import argparse
import numpy as np
import os
import cv2
import mediapipe as mp
import pickle
'''
    This module is developed based on:
        https://github.com/google/mediapipe
'''


def get_parser():
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--filename', default='./data/test.mp4')
    parser.add_argument('--outname', default='./data/sample.pkl')
    parser.add_argument('--num_frames', default=10, type=int)
    return parser

arg = get_parser().parse_args()
filename = arg.filename
outname = arg.outname
num_frames = arg.num_frames

mp_pose = mp.solutions.pose
# For webcam input:
cap = cv2.VideoCapture(filename)
pose_list = []

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    frames = 0
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break

        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks is not None and results.pose_landmarks.landmark is not None:
            single_pose = []
            for single_landmark in results.pose_landmarks.landmark:
                single_pose.append((single_landmark.x, single_landmark.y, single_landmark.z, single_landmark.visibility))
            pose_list.append(single_pose)
        if 0xFF == 27:  
            break
        frames = frames + 1
    cap.release()

# output pose
pose_array = np.array(pose_list)

if(len(pose_list) > num_frames):
    pose_array_list = []
    for i in range(0, pose_array.shape[0]-num_frames, 5):
      pose_array_list.append(pose_array[i : i + num_frames])
    pose_array_list = np.array(pose_array_list)
    pickle.dump(pose_array_list, open(outname, 'wb'))
    print(filename + ' finished; total frames:' + str(frames) + '; total pose: ' + str(len(pose_list)) + '; array shape: ' + str(pose_array_list.shape))
else:
    print('None has been extracted.')

