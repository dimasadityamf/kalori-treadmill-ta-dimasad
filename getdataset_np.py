# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 21:59:01 2023

@author: dimaz
"""

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import os
import cvlib as cv

import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# For webcam input:
#vid = cv2.VideoCapture(0)
vid = cv2.VideoCapture('video_train/kiri2_train.mp4')

Counter = 0

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while vid.isOpened():
    success, image = vid.read()
    ret, image = vid.read()
    
    if image is None:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      break
    
    image_height, image_width, _ = image.shape
    height, width, _ = image.shape

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    blank_image = np.zeros((image_height,image_width,3), np.uint8)
    
    # Bbox padd amount
    padd_amount = 5
    
    landmarks = []
    lines = []
    
    
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if not results.pose_landmarks:
      continue
    
    if results.pose_landmarks:
        
        #idx_line = [16, 14, 12, 11, 13, 15]
        #idx_line = [28, 26, 24, 23, 25, 27]
        idx_line = [28, 26, 25, 27]
        
        for i in idx_line:
            landmark = results.pose_landmarks.landmark[i]
            lines.append((int(landmark.x * width), int(landmark.y * height)))
            for index, item in enumerate(lines): 
                if index == len(lines) -1:
                    break
                
                cv2.line(blank_image, item, lines[index + 1], [255, 255, 255], 2)
                
        for i in idx_line:
            # Append the landmark into the list.
            landmark = results.pose_landmarks.landmark[i]
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))
                
            x = int(landmark.x * image_width)
            y = int(landmark.y * image_height)
            
            cv2.circle(blank_image, (x, y), 7, (255, 255, 255), 3)
            
            if i == 28:
                cv2.circle(blank_image, (x, y), 7, (19, 19, 191), -1)
            if i == 26:
                cv2.circle(blank_image, (x, y), 7, (255, 41, 41), -1)
            if i == 24:
                cv2.circle(blank_image, (x, y), 7, (40, 158, 30), -1)
            if i == 23:
                cv2.circle(blank_image, (x, y), 7, (224, 213, 16), -1)
            if i == 25:
                cv2.circle(blank_image, (x, y), 7, (16, 70, 224), -1)
            if i == 27:
                cv2.circle(blank_image, (x, y), 7, (204, 16, 224), -1)
            
        x_coordinates = np.array(landmarks)[:,0]
        
        y_coordinates = np.array(landmarks)[:,1]
        
        x1  = int(np.min(x_coordinates) - padd_amount)
        y1  = int(np.min(y_coordinates) - padd_amount)
        x2  = int(np.max(x_coordinates) + padd_amount)
        y2  = int(np.max(y_coordinates) + padd_amount)
        
        cv2.rectangle(image, (x1, y1), (x2, y2), (155, 0, 255), 3, cv2.LINE_8)
    
        # Croping as bounding box
        if y1<0:
            cropped_bbox = blank_image[0:y2, x1:x2]
        elif x1<0:
            cropped_bbox = blank_image[y1:y2, 0:x2]
        else:
            cropped_bbox = blank_image[y1:y2, x1:x2]
            
        resized_cropped_bbox = cv2.resize(cropped_bbox, (320, 320))
            
        Counter = Counter+1
        sfc = "export"+"\\"+str(Counter)
        filename=sfc+".jpg"
        cv2.imwrite(filename, resized_cropped_bbox)
    
    
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', image)
    cv2.imshow('landmark', blank_image)
    cv2.imshow('Cropped', resized_cropped_bbox)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
vid.release()
cv2.destroyAllWindows()