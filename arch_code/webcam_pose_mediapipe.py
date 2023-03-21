# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 12:31:08 2022

@author: eko my
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 21:01:20 2022

@author: dimaz
"""
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import os
import cvlib as cv

import cv2
import numpy as np
import mediapipe as mp
import time 

# load model
model = load_model('pose.model')


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
pose = mp_pose.Pose( static_image_mode=True,  model_complexity=2, enable_segmentation=True,min_detection_confidence=0.5)

# define a video capture object
vid = cv2.VideoCapture(0)

#vid = cv2.VideoCapture('pose.jpg')

#vid = cv2.VideoCapture('run.mp4')

classes = ['tpose','ypose']

TimeStart = time.time() 
TimeNow = time.time() 
Counter = 0
step = 0
walk = False
  
while vid.isOpened():
      
    # Capture the video frame
    # by frame
        
    #vid.set(3,1280)
    #vid.set(4,720)
    
    
    
    ret, image = vid.read()
    
    #face, confidence = cv.detect_face(image)
    
    
    
    image_height, image_width, _ = image.shape
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Find webcam resolution
    b_width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    b_height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    # Change value to int
    b_width = int(b_width)
    b_height = int(b_height)
    #print(b_width, b_height)
    
    # Create blank image/black frame with webcam resolution
    blank_image = np.zeros((b_height,b_width,3), np.uint8)
    

    # Retrieve the height and width of the input image.
    height, width, _ = image.shape
    
    # Bbox padd amount
    padd_amount = 5
    
    # Initialize a list to store the detected landmarks.
    landmarks = []
    
    # Create bounding box for skeleton
    if results.pose_landmarks:
        # Iterate over the detected landmarks.
        
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
        
        cv2.rectangle(blank_image, (x1, y1), (x2, y2), (155, 0, 255), 3, cv2.LINE_8)
        
        # Create landmarks
        mp_drawing.draw_landmarks(
            blank_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
        # Save image of blackscreen with skeleton as bounding box resolution
        
        
          
        # Croping as bounding box
        #print(x1, y1, x2, y2)
        if y1<0:
            cropped_bbox = blank_image[0:y2, x1:x2]
        elif x1<0:
            cropped_bbox = blank_image[y1:y2, 0:x2]
        else:
            cropped_bbox = blank_image[y1:y2, x1:x2]
        
        # Resized and write image
        #resized_cropped_bbox = cv2.resize(cropped_bbox, (320, 320))
        #cv2.imwrite(filename, resized_cropped_bbox)
        
        #for idx, f in enumerate(face):
        for idx  in range(0, 1):
            
            (startX, startY) = x1, y1
            (endX, endY) = x2, y2
        
            pose_crop = np.copy(blank_image[startY:endY,startX:endX])
            #cv2.imshow('pose_crop', pose_crop)
            cv2.rectangle(image, (startX,startY), (endX,endY), (0,255,0), 2)
            
            if (pose_crop.shape[0]) < 10 or (pose_crop.shape[1]) < 10:
                continue
            
            pose_crop = cv2.resize(pose_crop, (96,96))
            pose_crop = pose_crop.astype("float") / 255.0
            pose_crop = img_to_array(pose_crop)
            pose_crop = np.expand_dims(pose_crop, axis=0)
            
   
            conf = model.predict(pose_crop)[0]
            #print(conf)
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

          
          # Display the resulting frame
          #cv2.imshow('camera', image)
          #cv2.imshow('landmark', blank_image)
          #cv2.imshow('Cropped', resized_cropped_bbox)
        
    cv2.imshow('camera', image)
    cv2.imshow('landmark', blank_image)
    
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()