# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 09:42:45 2023

@author: dimaz
"""

        idx_line = [28, 26, 24, 23, 25, 27]
        
        for i in idx_line:
            # Append the landmark into the list.
            landmark = results.pose_landmarks.landmark[i]
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))
            lines.append((int(landmark.x * width), int(landmark.y * height)))
                        
            for index, item in enumerate(lines): 
                if index == len(lines) -1:
                    break
                cv2.line(blank_image, item, lines[index + 1], [255, 255, 255], 2) 
                
            x = int(landmark.x * image_width)
            y = int(landmark.y * image_height)
            
            if i == 28:
                cv2.circle(blank_image, (x, y), 7, (255, 90, 90), -1)
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
                            
            cv2.circle(blank_image, (x, y), 7, (255, 255, 255), 3)
            
            
            
            
            
        idx_line = [16, 14, 12, 11, 13, 15]
        
        for i in idx_line:
            # Append the landmark into the list.
            landmark = results.pose_landmarks.landmark[i]
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))
            lines.append((int(landmark.x * width), int(landmark.y * height)))
                        
            for index, item in enumerate(lines): 
                if index == len(lines) -1:
                    break
                cv2.line(blank_image, item, lines[index + 1], [255, 255, 255], 2) 
                
            x = int(landmark.x * image_width)
            y = int(landmark.y * image_height)
            
            cv2.circle(blank_image, (x, y), 7, (255, 255, 255), 3)
            cv2.circle(blank_image, (x, y), 7, (31, 136, 246), -1)