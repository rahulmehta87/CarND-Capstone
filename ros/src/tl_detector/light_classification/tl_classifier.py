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

#import yaml

#PATH_TF_MODELS_RESEARCH = "/home/student/github/models/research"
#PATH_TF_MODELS_SLIM = "/home/student/github/models/research/slim"
#PATH_TF_MODELS_OBJECT_DETECTION = "/home/student/github/models/research/object_detection"

#sys.path.append(PATH_TF_MODELS_RESEARCH)
#sys.path.append(PATH_TF_MODELS_SLIM)
#sys.path.append(PATH_TF_MODELS_OBJECT_DETECTION)
#from utils import label_map_util
#from utils import visualization_utils as vis_util

#from utils import dataset_util

FILE_PREFIX_IMG = "IMG_"
DIR_DATA = "DATA/"

#PATH_TEST_IMAGE_FILE = "/home/student/CarND-Capstone/ros/src/tl_detector/DATA/IMG_20180701_104354_0.png"

#PATH_TRAINED_GRAPH = "/home/student/github/models/research/object_detection/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb"
#PATH_TRAINED_GRAPH = "/home/student/CarND-Capstone/ros/src/tl_detector/frozen_inference_graph.pb"

#PATH_LABEL_MAP = "/home/student/github/models/research/object_detection/data/mscoco_label_map.pbtxt"
#NUM_CLASSES = 90

#PATH_TRAIN_DATA_DIR = ""
#PATH_YAML = "/home/student/Downloads/train.yaml" #TODO change the path
#WIDTH_TRAIN_DATA = 1280
#HEIGHT_TRAIN_DATA = 720
#NUM_CLASSES_TRAIN_DATA = 14
#PATH_TF_RECORD = "train_data.tfrecords"
#DICT_LABEL = { "Green" : 1, "Red" : 2, "GreenLeft" : 3, "GreenRight" : 4,
#    "RedLeft" : 5, "RedRight" : 6, "Yellow" : 7, "off" : 8,
#    "RedStraight" : 9, "GreenStraight" : 10, "GreenStraightLeft" : 11, "GreenStraightRight" : 12,
#    "RedStraightLeft" : 13, "RedStraightRight" : 14 }
DICT_LABEL = { "green" : 1, "red" : 2, "yellow" : 3, "off" : 4 }

class TLClassifier(object):
    def __init__(self):
        #self.load_model(PATH_TRAINED_GRAPH_SITE)
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


#    def run_detection(self, image):
#        rospy.loginfo('[KONO] CALLED run_detection')
#        with self.detection_graph.as_default():
#            with tf.Session(graph=self.detection_graph) as sess:
#                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
#                detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
#                detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
#                detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
#                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
#                (boxes, scores, classes, num) = sess.run(
#                    [detection_boxes, detection_scores, detection_classes, num_detections],
#                    feed_dict={image_tensor: image})
#                return (boxes, scores, classes, num)

    def run_detection(self, image):
        with self.detection_graph.as_default():
            (boxes, scores, classes, num) = self.sess.run(
                [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: image})
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)
        return (boxes, scores, classes, num)

    def judge_traffic_light(self, scores, classes, num):
        #rospy.loginfo('[KONO] CALLED judge_traffic_light')
        #rospy.loginfo('[KONO] len of classes: %s', len(classes))
        #rospy.loginfo('[KONO] len of scores: %s', len(scores))
        #rospy.loginfo('[KONO] num: %s', num)

        high_score = 0
        class_label = None
        THRES_SCORE = 0.5
        #for i in range(num):
        #    rospy.loginfo('[KONO] classes[i]: %s', classes[i])
        #    rospy.loginfo('[KONO] scores[i]: %s', scores[i])
        #    if (float(scores[i]) > high_score):
        #        high_score = scores[i]
        #        class_label = int(classes[i])
        #        if (high_score > THRES_SCORE):
        #            break
        rospy.loginfo('[KONO] classes[0]: %s', classes[0])
        rospy.loginfo('[KONO] scores[0]: %s', scores[0])
        high_score = scores[0]
        class_label = int(classes[0])
        if (high_score > THRES_SCORE):
        #if ((not class_name is None) & high_score > THRES_SCORE):
             return self.convert_class(class_label)
        else:
             return TrafficLight.UNKNOWN

    def convert_class(self, class_name):
        #rospy.loginfo('[KONO] CALLED convert_class')
        if class_name == 1: #in (1, 'green'):
            return TrafficLight.GREEN
        elif class_name == 2: #in (2, 'red'):
            return TrafficLight.RED
        elif class_name == 3: #in (3, 'yellow'):
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
        #TODO implement light color prediction
        #rospy.loginfo('[KONO] CALLED get_classification')
        feed_image = np.expand_dims(image, axis=0)
        (boxes, scores, classes, num) = self.run_detection(feed_image)
        #tl_color = self.judge_traffic_light(scores[0], classes[0], num[0])
        tl_color = self.judge_traffic_light(scores, classes, num)
        #rospy.loginfo('[KONO] tl_color: %s', tl_color)
        return tl_color
        #return TrafficLight.UNKNOWN

    def save_training_data(self, cv_image, label):
        path = DIR_DATA + FILE_PREFIX_IMG + "{0:%Y%m%d_%H%M%S}_{1}.png".format(datetime.datetime.now(), label)
        cv2.imwrite(path, cv_image)

#    def generate_model(self):
#        pass


#if __name__ == '__main__':
#    light_classifier = TLClassifier()
#    light_classifier.test() # generate_model()
#    if (not os.path.exists(PATH_TF_RECORD)):
#        light_classifier.write_tf_record()
