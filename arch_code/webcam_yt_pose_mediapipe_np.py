# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 20:00:44 2023

@author: dimaz
"""

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import os
import cvlib as cv

import cv2
import numpy as np
import mediapipe as mp

# load model
model = load_model('yt_pose.model')

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# For webcam input:
vid = cv2.VideoCapture(0)

classes = ['tpose','ypose']

step = 0
walk = False

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while vid.isOpened():
    success, image = vid.read()
    ret, image = vid.read()
    
    face, confidence = cv.detect_face(image)
    
    image_height, image_width, _ = image.shape
    height, width, _ = image.shape
    
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    blank_image = np.zeros((image_height,image_width,3), np.uint8)
    
    # Bbox padd amount
    padd_amount = 5
    
    landmarks = []
    
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if not results.pose_landmarks:
      continue
    
    if results.pose_landmarks:
        
        for landmark in results.pose_landmarks.landmark:
            # Append the landmark into the list.
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))
            
            
        #print(landmarks)    
        x_coordinates = np.array(landmarks)[:,0]
        
        y_coordinates = np.array(landmarks)[:,1]
        
        x1  = int(np.min(x_coordinates) - padd_amount)
        y1  = int(np.min(y_coordinates) - padd_amount)
        x2  = int(np.max(x_coordinates) + padd_amount)
        y2  = int(np.max(y_coordinates) + padd_amount)
        
        #cv2.rectangle(blank_image, (x1, y1), (x2, y2), (155, 0, 255), 3, cv2.LINE_8)
        
        # Create landmarks
        mp_drawing.draw_landmarks(
            blank_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
        # Croping as bounding box
        if y1<0:
            cropped_bbox = blank_image[0:y2, x1:x2]
        elif x1<0:
            cropped_bbox = blank_image[y1:y2, 0:x2]
        else:
            cropped_bbox = blank_image[y1:y2, x1:x2]
            
        #resized_cropped_bbox = cv2.resize(cropped_bbox, (320, 320))
        
        
        for idx  in range(0, 1):
            print(idx)
            # get corner points of face rectangle        
            (startX, startY) = x1, y1
            (endX, endY) = x2, y2
    
            # draw rectangle over face
            # cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)
    
            # crop the detected face region
            pose_crop = np.copy(blank_image[startY:endY,startX:endX])
            #cv2.imshow('pose_crop', pose_crop)
            cv2.rectangle(image, (startX,startY), (endX,endY), (0,255,0), 2)
            
            if (pose_crop.shape[0]) < 10 or (pose_crop.shape[1]) < 10:
                continue
            
            pose_crop = cv2.resize(pose_crop, (96,96))
            pose_crop = pose_crop.astype("float") / 255.0
            pose_crop = img_to_array(pose_crop)
            pose_crop = np.expand_dims(pose_crop, axis=0)
            
   
            conf = model.predict(pose_crop)[0] # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]
    
            # get label with max accuracy
            idx = np.argmax(conf)
            label = classes[idx]
    
            label = "{}: {:.2f}%".format(label, conf[idx] * 100)
    
            Y = startY - 10 if startY - 10 > 10 else startY + 10
            
            if idx == 0:
                walk = False
            if idx == 1 and walk == False:
                walk = True
                step +=1
            
            # Setup status box
            cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
                  
            cv2.putText(image, label, 
                        (10,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(image, str(step), 
                        (13,60), 
                        cv2.FONT_HERSHEY_SIMPLEX,0.8, (255,255,255), 2, cv2.LINE_AA) 
        
    
    
    
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', image)
    cv2.imshow('landmark', blank_image)
    #cv2.imshow('Cropped', resized_cropped_bbox)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
vid.release()
cv2.destroyAllWindows()