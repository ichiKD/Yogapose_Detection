import cv2
import numpy as np
import os
for x in range(1, 12):
    input_folder = r'/home/20ucc067/Shubhang/DATASET_MKV//'+str(x)
    #input_folder = r'C:\Users\shubh\Desktop\Rishikesh\DATASET_MKV\1'
    input_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.mkv')]
    target_height = 128
    target_width = 128
    frames = []
    for input_file in input_files:
        input_path = os.path.join(input_folder, input_file)
        video_capture = cv2.VideoCapture(input_path)
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            if frame.shape[0] != target_height or frame.shape[1] != target_width:
                frame = cv2.resize(frame, (target_width, target_height))
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray_frame)
        video_capture.release()
    frames_array = np.array(frames)
    #output_path = os.path.join(input_folder, str(x)+'.npy')
    output_path = os.path.join(r'/home/20ucc067/Shubhang/FINAL_NPY//', str(x)+'.npy')
    np.save(output_path, frames_array)
    print(x, "is saved")