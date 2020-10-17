import cv2
from PIL import Image
import pytesseract
import numpy as np
import multiprocessing as mp

def get_strings_multiprocessing(frames):
    num_processes = mp.cpu_count()
    print(f"num_processes = {num_processes}")
    p = mp.Pool(num_processes)
    fields = p.map(get_string, frames)
    return fields

def get_strings(frames):
    fields = []
    for frame in frames:
        fields.append(get_string(frame))
    return fields

def get_string(frame):
    h,w = frame.shape[:2]
    h = int(h/6); w = int(w/2)
    frame = frame[:h,:w,:]
    # grey-scale the frames (green=255,else=0)
    frame = 255*np.where(frame[:,:,0]==0, 1, 0)*np.where(frame[:,:,1]==255, 1, 0)*np.where(frame[:,:,2]==0, 1, 0)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Do dilation and erosion to eliminate unwanted noises
    kernel = np.ones((1, 1), np.uint8)
    frame = cv2.dilate(frame, kernel, iterations=20)
    frame = cv2.erode(frame, kernel, iterations=20)
    # convert cv2 img to PIL img
    frame_RGB = cv2.cvtColor(frame.astype("uint8"), cv2.COLOR_GRAY2RGB)
    PIL_frame = Image.fromarray(frame_RGB)
    # Character recognition with tesseract
    string = pytesseract.image_to_string(PIL_frame)
    string = float(string.replace(" ","").split("(Oe)")[0])
    return string