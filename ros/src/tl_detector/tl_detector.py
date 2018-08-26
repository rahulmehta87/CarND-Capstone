#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
from scipy.spatial import KDTree
import datetime

import threading

STATE_COUNT_THRESHOLD = 3

FLG_TRAINNING_DATA_COLLECTION = False #True
#FLG_USE_GROUND_TRUTH = True # False
FLG_USE_GROUND_TRUTH = False

LOOKAHEAD_WPS = 25 # 50 # Number of waypoints we will publish. You can change this number

PATH_TRAINED_GRAPH_SIM = "./frozen_inference_graph.pb"
PATH_TRAINED_GRAPH_SITE = "./site/frozen_inference_graph.pb"

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        self.waypoints_2d = None
        self.waypoint_tree = None

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        flg_site = False
        if ('is_site' in self.config):
            flg_site = self.config['is_site']
        if FLG_USE_GROUND_TRUTH == False:
            if flg_site == True:
                self.light_classifier.load_model(PATH_TRAINED_GRAPH_SITE)
            else:
                self.light_classifier.load_model(PATH_TRAINED_GRAPH_SIM)

        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        self.threading_rlock = threading.RLock()

        rospy.spin()
        #self.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        if self.threading_rlock.acquire(True):
            self.has_image = True
            self.camera_image = msg
            #light_wp, state = self.process_traffic_lights()

            '''
            Publish upcoming red lights at camera frequency.
            Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
            of times till we start using it. Otherwise the previous stable state is
            used.
            '''
            light_wp, state = self.process_traffic_lights()
            if self.state != state:
                self.state_count = 0
                self.state = state
            elif self.state_count >= STATE_COUNT_THRESHOLD:
                self.last_state = self.state
                light_wp = light_wp if state == TrafficLight.RED else -1
                self.last_wp = light_wp
                self.upcoming_red_light_pub.publish(Int32(light_wp))
            else:
                self.upcoming_red_light_pub.publish(Int32(self.last_wp))
            self.state_count += 1

            self.threading_rlock.release()

    """
    def spin(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            while self.pose is not None and self.waypoints is not None and self.camera_image is not None:
                light_wp, state = self.process_traffic_lights()
                if self.state != state:
                    self.state_count = 0
                    self.state = state
                elif self.state_count >= STATE_COUNT_THRESHOLD:
                    self.last_state = self.state
                    light_wp = light_wp if state == TrafficLight.RED else -1
                    self.last_wp = light_wp
                    self.upcoming_red_light_pub.publish(Int32(light_wp))
                else:
                    self.upcoming_red_light_pub.publish(Int32(self.last_wp))
                self.state_count += 1

            rate.sleep()
    """

    #def get_closest_waypoint(self, pose):
    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]
        return closest_idx #0

    def is_near_by_traffic_light(self):
        closest_idx = 0;
        if(self.pose):
            closest_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)
        #closest_idx = self.get_closest_waypoint_idx()
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        #return self.stopline_wp_idx == -1 or (self.stopline_wp_idx >= farthest_idx)
        return self.last_wp == -1 or (self.last_wp >= farthest_idx)

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        #path = "light_classification/DATA/{0:%H%M%S}_{1}.png".format(datetime.datetime.now(), light.state)
        #cv2.imwrite(path, cv_image)

        if FLG_TRAINNING_DATA_COLLECTION:
            label = TrafficLight.UNKNOWN
            if self.is_near_by_traffic_light():
                label = light.state
            self.light_classifier.save_training_data(cv_image, label)

        #Get classification
        #_ = self.light_classifier.get_classification(cv_image)
        # For testing, just return the light state
        # TODO:

        #return tl_class
        # Use Ground Truth
        if FLG_USE_GROUND_TRUTH:
            #rospy.loginfo('[KONO] GROUND TRUTH light.state :%s', light.state)
            return light.state
        else:
            tl_class = self.light_classifier.get_classification(cv_image)
            rospy.loginfo('[KONO] tl_class: %s light.state :%s', tl_class, light.state)
            return tl_class

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        closest_light = None
        line_wp_idx = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            #car_position = self.get_closest_waypoint(self.pose.pose)
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)

            #TODO find the closest visible traffic light (if one exists)
            diff = len(self.waypoints.waypoints)
            for i, light in enumerate(self.lights):
                # Get stop line waypoint index
                line = stop_line_positions[i]
                temp_wp_idx = self.get_closest_waypoint(line[0], line[1])
                # Find closest stop line waypoint index
                d = temp_wp_idx - car_wp_idx
                if d >= 0 and d < diff:
                    diff = d
                    closest_light = light
                    line_wp_idx = temp_wp_idx

        if closest_light:
            state = self.get_light_state(closest_light)
            #rospy.loginfo('[KONO] line_wp_idx: %s, state: %s', line_wp_idx, state)
            return line_wp_idx, state
        # As there is no line below in Walkthrough code, I comment out for now.
        #self.waypoints = None
        rospy.loginfo('closest_light is False')
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
