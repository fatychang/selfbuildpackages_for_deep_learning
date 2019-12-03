# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 16:31:46 2019

This class is responsible for detecting object(s) in the image or frame.
It initialize the network, blob, setInput and run the prediction etc.

@author: jschang
"""

import cv2
import numpy as np


class SSD:
    
    def __init__(self, confidence=0.5): 
        # initialize the ssd class 
        # and set the confidence level
        self.path_prototxt = "D:\\Jen\\_Documents\\eLearning\\Computer Vision\\pyimagesearch\\real-time-object-detection\\MobileNetSSD_deploy.prototxt.txt"
        self.path_model = "D:\\Jen\\_Documents\\eLearning\\Computer Vision\\pyimagesearch\\real-time-object-detection\\MobileNetSSD_deploy.caffemodel"
        self.confidence = confidence
        self.net = None
        self.labels = None
        self.colors = None
        self.blob = None
        
        self.h = None
        self.w = None
        
        self.COLORS = []
        self.box_center = []
        self.ignored = []

            
    def start(self):
        # return self 
        return self
    
    def load_classes(self):
        # load the pre-defined labels
        self.labels =  ["background", "aeroplane", "bicycle", "bird", "boat",
                	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                	"sofa", "train", "tvmonitor"]
        return self.labels
    
    def generate_colors(self, length):
        # generate randon color list for each class
        self.COLORS = np.random.randint(0, 255, (length, 3), dtype="uint8")
        
        return self.colors
    
    
    def load_model(self):
        # load the model from local disk 
        # and return the net
        self.net = cv2.dnn.readNetFromCaffe(self.path_prototxt, self.path_model)
        
    def blob_image(self, image):
        # convert the image to a blob
        # and get the dimensions of the image
        (self.h, self. w) = image.shape[:2]
        self.blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 
                                     0.007843, (300, 300), 127, 5)
        return self.blob
        
    def get_detections(self, blob):
        # set the network input 
        # and pass input through the network
        self.net.setInput(blob)
        return self.net.forward()
    
    def get_confidence(self, detections, idx):
        # return the confidence of the assigned detection
        return detections[0, 0, idx, 2]
    
    def get_class_idx(self, detections, idx):
        # return the index of the class for the assigned detection
        return int(detections[0, 0, idx, 1])
    
    def get_class_label(self, idx):
        # return the label of the assigned detection
        return self.labels[idx]
    
    def get_bounding_box(self, detections, idx):
        # return the bonding box in the order of
        # (startX, startY, endX, endY)
        return (detections[0, 0, idx, 3:7] * np.asarray(
                [self.w, self.h, self.w, self.h])).astype("int")
        
    def get_color(self, idx):
        # return the corresponding color
        return [int (c) for c in self.COLORS[idx]]
    
    def draw_bounding_box(self, image, detections, idx, COLORS):
        # draw the bounding box on image and its descriptions
        # draw the center point of the box
        confidence = self.get_confidence(detections, idx)
        class_idx = self.get_class_idx(detections, idx)
        label = self.get_class_label(class_idx)
        color = self.get_color(class_idx)
        (startX, startY, endX, endY) = self.get_bounding_box(detections, idx)
        
        text = "{}: {:.2f}%".format(label, confidence * 100)
        text_y = startY-15 if startY - 15 > 15 else startY + 15
        
        cv2.rectangle(image, (startX, startY),(endX, endY), color, 2)
        cv2.putText(image, text, (startX, text_y),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2)
    
        # get the center point of the bounding box
        box_center_x = int((startX + endX)/2)
        box_center_y = int((startY + endY)/2)
        self.box_center=[box_center_x, box_center_y]
        
        # draw the point representing the center of the box
        cv2.circle(image, (box_center_x, box_center_y), 1, color, 2)
        
        
    def get_box_center(self):
        # return the position of the center of the box
        return self.box_center
    
    def clear_box_center(self):
        # clear the list storing the position of the center of the box
        self.box_center = []
         
    def set_model_prototext_path(self, path):
        # set the prototxt path if it's not at default location
        self.path_prototxt = path
        
    def set_model_path(self, path):
        # set the model path if it's not at default location
        self.path_model = path
    
    def set_ignored_classes(self, ignored):
        self.ignored = ignored
        