#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 23:07:46 2021

@author: kesaprm
"""

import cv2
from tracker import *
import matplotlib.pyplot as plt


cap = cv2.VideoCapture("M0 16H 5m 10x.avi")
# Create tracker object
tracker = EuclideanDistTracker()
frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4)) 
size = (frame_width, frame_height) 

obj_detect = cv2.createBackgroundSubtractorMOG2(history = 100,varThreshold =10)
out = cv2.VideoWriter('filename.avi',  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         10, size) 
while True:
    ret, frame = cap.read()
    
    #1. Object detection
    mask = obj_detect.apply(frame)
    _,mask = cv2.threshold(mask,254,255,cv2.THRESH_BINARY)
    contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections =[]
    for cnt in contours:
        # remove small elements
        area = cv2.contourArea(cnt)
        if area > 300:
            cv2.drawContours(frame, [cnt], -1, (0,255,0),2)
            x,y,w,h = cv2.boundingRect(cnt)
            #cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),3)
            
            detections.append([x,y,w,h])
            
    #2. Object tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(frame, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)


    # write the flipped frame
    #out.write(frame)

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # else:
    #     break
    cv2.imshow("Frame",frame)
    #cv2.imshow("Mask",mask)
    key = cv2.waitKey(30)
    if key == 27:
        break
    

cap.release()
out.release()
cv2.destroyAllWindows()

