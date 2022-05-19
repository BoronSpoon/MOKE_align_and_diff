import cv2
import numpy as np 
import sys
import math

def open_video(path):
    cap = cv2.VideoCapture(path)
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    frames = []
    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            frames.append(frame)
        else:
            break
    cap.release()
    return frames

def split_odd_even_frames(frames):
    odd_frames = frames[0::2] # odd: 1,3,5,...
    even_frames = frames[1::2] # even: 2,4,6,...
    return odd_frames, even_frames

def write_odd_even_frames(odd_frames, even_frames, odd_path, even_path):
    for frames, path in [[odd_frames,odd_path],[even_frames,even_path]]:
        writer = cv2.VideoWriter(path,  
                                cv2.VideoWriter_fourcc(*'MJPG'), 
                                10, frames[0].shape[:2][::-1])
        for frame in frames:
            writer.write(frame)
        writer.release()

if __name__ == "__main__":
    path = sys.argv[1] # 動画のパス
    odd_path = path.replace(".avi","_odd.avi") # odd frame avi path
    even_path = path.replace(".avi","_even.avi") # even frame avi path

    frames = open_video(path) # 動画の読み込み
    odd_frames, even_frames = split_odd_even_frames(frames)
    write_odd_even_frames(odd_frames, even_frames, odd_path, even_path)
    
    #.destroyAllWindows() 
    print("The video was successfully saved") 
