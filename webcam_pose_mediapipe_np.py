# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 20:00:44 2023

@author: dimaz
"""

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import os
import cvlib as cv
import math
from math import hypot

import cv2
import numpy as np
import mediapipe as mp

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
df = pd.read_csv('regresi/dataset_panjang_v2.csv')

# load model
model = load_model('model/step_kaki_samping3.model')

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# For webcam input:
#vid = cv2.VideoCapture(0)
vid = cv2.VideoCapture('video_treadmill/speed_12_fps25.mp4')

classes = ['kanan','kiri']

step = 0
walk = False
i_idx = 0
iteration_idx = 2
arr_idx = [0] * iteration_idx
cframe = 0
result = False
i_step = 0
iteration_step = 35
arr_step = [0] * iteration_step
is_step = False

hrs = 0
mins = 0
sec = 0
period = '00:00:00'

frame_width = int(vid.get(3))
frame_height = int(vid.get(4))

def reg_panjang(ac_time, step):
    poly = PolynomialFeatures(degree = 4)
    X_poly = poly.fit_transform(df.drop('panjang',axis='columns'))
    
    poly.fit(X_poly, df.panjang)
    lin2 = linear_model.LinearRegression()
    lin2.fit(X_poly, df.panjang)

    pred2array = np.array([[ac_time, step]])
    reg_pred = lin2.predict(poly.fit_transform(pred2array))
    
    return reg_pred[0]


with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while vid.isOpened():
    success, image = vid.read()
    ret, image = vid.read()
    
    if image is None:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      result = True
      break
  
    image = cv2.resize(image, (960,540))
    #image = cv2.flip(image,1)
    
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
    
    
    if sec > 59:
        sec = 0
        mins = mins+1

    if mins > 59:
        mins = 0
        hrs = hrs+1

    if hrs > 23:
        hrs = 0

    period = "{:02d}:{:02d}:{:02d}".format(hrs,mins,sec)

    
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
        
        #mp_drawing.draw_landmarks(
        #    image,
        #    results.pose_landmarks,
        #    mp_pose.POSE_CONNECTIONS,
        #    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        
        #cv2.rectangle(blank_image, (x1, y1), (x2, y2), (155, 0, 255), 3, cv2.LINE_8)
    
        # Croping as bounding box
        if y1<0:
            cropped_bbox = blank_image[0:y2, x1:x2]
        elif x1<0:
            cropped_bbox = blank_image[y1:y2, 0:x2]
        else:
            cropped_bbox = blank_image[y1:y2, x1:x2]
            
        #resized_cropped_bbox = cv2.resize(cropped_bbox, (320, 320))
        
        
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
        print(conf)
        # get label with max accuracy
        idx = np.argmax(conf)
        label = classes[idx]
        print(idx)
        print(step)
        label = "{}: {:.2f}%".format(label, conf[idx] * 100)
        t_step = "step: {}".format(str(step))

        
        
        landdiff = lambda p1, p2: (p1[0]-p2[0], p1[1]-p2[1])
        diffs = (landdiff(p1, p2) for p1, p2 in zip (landmarks, landmarks[1:]))
        path = sum(hypot(*d) for d in  diffs)
        print(path)
        
        act_size = 1.0 / path
        
        lx1 = results.pose_landmarks.landmark[28].x
        ly1 = results.pose_landmarks.landmark[28].y
        lx2 = results.pose_landmarks.landmark[27].x
        ly2 = results.pose_landmarks.landmark[27].y
        dist = math.sqrt( (lx2 - lx1)**2 + (ly2 - ly1)**2 )

        panjang = dist * act_size
        print(panjang)


        Y = startY - 10 if startY - 10 > 10 else startY + 10

        if idx == 0 and walk == True:
            if (1 not in arr_idx):
                walk = False
                step +=1
                i_idx = 0
            
        if idx == 1 and walk == False:
            if (0 not in arr_idx):
                walk = True
                step +=1
                i_idx = 0
            
        if i_idx == iteration_idx:
            i_idx = 0
        
        print(i_idx)
        arr_idx[i_idx] = idx
        print(arr_idx)
        i_idx += 1
        
        if i_step == iteration_step:
            i_step = 0
        
        if step > 0:
            is_step = True
        
        arr_step[i_step] = step
        i_step += 1
        print(arr_step)
        
        
        if is_step and step > 5:
            check_step = all(element == arr_step[0] for element in arr_step)
            if check_step:
                result = True
                break
        
        # Setup status box
        cv2.rectangle(image, (0,0), (225,125), (245,117,16), -1)
              
        cv2.putText(image, label, 
                    (10,30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        
        cv2.putText(image, t_step, 
                    (10,70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA) 
        
        cv2.putText(image, period,
                    (10,110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA) 
        
        
        cframe += 1
        if cframe == 13:
            if is_step:
                sec = sec + 1

            print("=================================================")
            cframe = 0
            
        #if sec > 0 and (sec-3):
        #    if step_sec == step:
        #        continue
        
        print(cframe)
        
    
    
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', image)
    cv2.imshow('landmark', blank_image)
    #cv2.imshow('Cropped', resized_cropped_bbox)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        result = True 
        break
    
  vid.release() 
  cv2.destroyAllWindows()  
  result_image = np.zeros((540,960,3), np.uint8)
  while result == True:       
      
      cv2.imshow('Result', result_image)
      
      ac_time = sec + (mins * 60) + (hrs * 24)
      
      #step_len = -0.00307786*ac_time + 0.00143617*float(step) + 0.8343048305325581
      #step_len = 0.6158285849
      
      step_len = reg_panjang(ac_time, float(step))
      
      distance = float(step) * step_len
      distance = round(distance, 3)

      if distance > 1000:
          distance = distance / 1000
          distance = round(distance, 3)
          u_distance = "km"
      else:
          u_distance = "m"

      reg_cal = -0.00030418*ac_time + 0.0704*distance -0.06703071381901893
      reg_cal = round(reg_cal, 3)
      
      t_distance = "distance: {} {}".format(distance, u_distance)
      r_time = "Time: {}".format(period)
      cal = "Calories: {}".format(reg_cal)

      cv2.putText(result_image, t_step, 
                  (10,60), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA) 
      
      cv2.putText(result_image, t_distance,
                  (10,100), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)
      
      cv2.putText(result_image, r_time,
                  (10,140), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)
      
      cv2.putText(result_image, cal, 
                  (10,180), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA) 

      if cv2.waitKey(1) & 0xFF == ord('q'):
          result = False
          break
        
vid.release()
cv2.destroyAllWindows()