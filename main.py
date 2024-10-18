##################################################################################
# Import the required libraries
##################################################################################
import pandas as pd
import numpy as np
import os
import sys
from ultralytics import YOLO
import random
import time
from tracker import Tracker
import cv2

##################################################################################
# set the working directory
##################################################################################
directory_path = '/Users/martinsegovia/Desktop/Object_Detection_and_Tracking'

os.chdir(directory_path)

print("Current Working Directory:", os.getcwd())

##################################################################################
# Read the video file
##################################################################################
# Input video
video_path = os.path.join('.','data', 'video', 'input', 'parkinglot.MOV')
mask_path = os.path.join('.','data', 'video', 'input', 'parkinglot_mask.jpg')

cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()
mask = cv2.imread(mask_path)

# Output video
video_output_path = os.path.join('.', 'data', 'video', 'output','parkinglot_output.mp4')

cap_out = cv2.VideoWriter(
    video_output_path,
    cv2.VideoWriter_fourcc(*'MP4V'),
    cap.get(cv2.CAP_PROP_FPS),
    (frame.shape[1], frame.shape[0])
)

##################################################################################
# Create the YOLO model for detection
##################################################################################
model = YOLO('yolo_model/yolov8m.pt')

##################################################################################
# Start the DeepSort tracker
##################################################################################
tracker = Tracker()

##################################################################################
# Process the video frame by frame
##################################################################################
detection_threshold = 0.8
time_threshold = 40

tracked_ids = {}

while ret:
    imgRegion = cv2.bitwise_and(frame, mask)

    # results = model(frame)
    results = model(imgRegion)
    
    current_time = time.time()

    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            if score > detection_threshold:
                detections.append([int(x1), int(y1), int(x2), int(y2), score])

        
        tracker.update(imgRegion, detections)
        # tracker.update(frame, detections)

        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            track_id = track.track_id

            if track_id not in tracked_ids:
                tracked_ids[track_id] = current_time

            # Calculate duration since the object was detected
            duration = current_time - tracked_ids[track_id]

            print(f"Processing Track ID={track_id}, Duration={duration:.2f} seconds")

            # Draw bounding box and display track ID and predicted class
            class_name = model.names[int(class_id)]  # Get class name from model

            # Determine rectangle color based on duration
            color = (0, 255, 0) if duration < time_threshold else (0, 0, 255)  # Green if < 10s, Red otherwise
            line_width = 1 if duration < time_threshold else 3


            cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_width)
            cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(frame, f'Time: {int(duration)}s', (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                
    print(tracked_ids)

    # Number of detected objects
    num_objects = len(tracked_ids)

    # Define board dimensions and position
    board_width, board_height = 250, 60
    top_left = (10, 10)
    bottom_right = (top_left[0] + board_width, top_left[1] + board_height)

    # Draw a semi-transparent rectangle for the board
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, (0, 0, 0), -1)  # Black background

    alpha = 0.6  # Transparency factor
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Add text displaying the number of objects detected
    text = f'Objects Detected: {num_objects}'
    cv2.putText(frame, text, (top_left[0] + 10, top_left[1] + 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


    # cv2.imshow("frame", frame)
    # if cv2.waitKey(25) & 0xFF == ord('q'):
    #     break

    cap_out.write(frame)
    ret, frame = cap.read()


cap.release()
cap_out.release()
cv2.destroyAllWindows()
  
  

