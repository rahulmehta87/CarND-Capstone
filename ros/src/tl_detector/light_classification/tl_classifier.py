from styx_msgs.msg import TrafficLight
import datetime
import cv2
import sys
import os
import rospy

import tensorflow as tf
import numpy as np
from PIL import Image

tf.logging.set_verbosity(tf.logging.ERROR)

FILE_PREFIX_IMG = "IMG_"
DIR_DATA = "DATA/"

DICT_LABEL = { "green" : 1, "red" : 2, "yellow" : 3, "off" : 4 }

class TLClassifier(object):
    def __init__(self):
        pass


    def load_model(self, path_to_trained_graph):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_to_trained_graph, 'rb') as fid:
                serialized_graph = fid.read()
                graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(graph_def, name='')

            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        self.sess = tf.Session(graph=self.detection_graph)


    def run_detection(self, image):
        with self.detection_graph.as_default():
            (boxes, scores, classes, num) = self.sess.run(
                [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: image})
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)
        return (boxes, scores, classes, num)

    def judge_traffic_light(self, scores, classes):
        THRES_SCORE = 0.5
        rospy.loginfo('[KONO] classes[0]: %s', classes[0])
        rospy.loginfo('[KONO] scores[0]: %s', scores[0])
        high_score = scores[0]
        class_label = int(classes[0])
        if (high_score > THRES_SCORE):
             return self.convert_class(class_label)
        else:
             return TrafficLight.UNKNOWN

    def convert_class(self, class_name):
        if class_name == 1:
            return TrafficLight.GREEN
        elif class_name == 2:
            return TrafficLight.RED
        elif class_name == 3:
            return TrafficLight.YELLOW
        else:
            return TrafficLight.UNKNOWN


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        feed_image = np.expand_dims(image, axis=0)
        (boxes, scores, classes, num) = self.run_detection(feed_image)
        tl_color = self.judge_traffic_light(scores, classes)
        return tl_color

    def save_training_data(self, cv_image, label):
        path = DIR_DATA + FILE_PREFIX_IMG + "{0:%Y%m%d_%H%M%S}_{1}.png".format(datetime.datetime.now(), label)
        cv2.imwrite(path, cv_image)

