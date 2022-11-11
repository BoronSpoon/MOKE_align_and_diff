import cv2
from PIL import Image
#import pytesseract
import numpy as np
import multiprocessing as mp

bounding_rectangle_shapes = { # fill_ratio, width, height
    (255.0, 9 , 10): ".",
    (255.0, 19, 9 ): "-",
    (142.0, 34, 55): "0",
    (130.3, 22, 54): "1",
    (136.1, 34, 54): "2",
    (121.9, 34, 55): "3",
    (119.1, 38, 53): "4",
    (140.8, 35, 54): "5",
    (145.3, 35, 55): "6",
    (99.20, 34, 53): "7",
    (147.5, 34, 55): "8",
    (146.9, 35, 55): "9",
}

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
    #cv2.imshow("ocr",frame.astype("uint8"))
    #cv2.waitKey(3000)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Do dilation and erosion to eliminate unwanted noises
    kernel = np.ones((1, 1), np.uint8)
    frame = cv2.dilate(frame, kernel, iterations=20)
    frame = cv2.erode(frame, kernel, iterations=20)

    # delete the following (Oe) by limiting the image size by finding the overall contour
    ret, frame_binary = cv2.threshold(frame.astype(np.uint8), 127, 255, 0)
    cnts, _ = cv2.findContours(frame_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_combined = np.concatenate(cnts)
    x, y, w, h = cv2.boundingRect(cnts_combined)
    max_x = x + w # position of the most right position 
    frame = frame[15:85,5:max_x-142] # remove the black space and "(Oe)"

    # find contour of each character
    ret, frame_binary = cv2.threshold(frame.astype(np.uint8), 127, 255, 0)
    cnts = list(cv2.findContours(frame_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
    cnts = [cv2.boundingRect(cnt) for cnt in cnts] # [x, y, w, h]
    
    #frame = cv2.merge([frame, frame, frame])
    #print(frame.shape)
    cnts = sorted(cnts, key=lambda cnt: cnt[0]) # sort from left to right
    string = ""
    
    #frame = cv2.resize(frame.astype(np.float32), None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST) # upconvert
    #frame = cv2.resize(frame.astype(np.float32), None, fx=5, fy=5, interpolation=cv2.INTER_LINEAR) # upconvert
    #frame = cv2.threshold(frame, 127, 255, 0)[1]*255
    # convert cv2 img to PIL img
    #frame_RGB = cv2.cvtColor(frame.astype("uint8"), cv2.COLOR_GRAY2RGB)
    #PIL_frame = Image.fromarray(frame_RGB)
    # Character recognition with tesseract
    #string = pytesseract.image_to_string(PIL_frame, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789.-') # whitelist characters
    #string = pytesseract.image_to_string(PIL_frame)
    try:
        for cnt in cnts:
            x, y, w, h = cnt
            #frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            frame_RGB = cv2.cvtColor(frame[y:y+h, x:x+w].astype("uint8"), cv2.COLOR_GRAY2RGB)
            PIL_frame = Image.fromarray(frame_RGB)
            frame_key = np.array([np.mean(frame[y:y+h, x:x+w]), w, h])
            square_error_key = np.array([np.sum((np.array(key) - frame_key)**2) for key in bounding_rectangle_shapes.keys()]) # get square sum error
            closest_key = list(bounding_rectangle_shapes.keys())[np.argmin(square_error_key)] # key with closest value
            char = bounding_rectangle_shapes[closest_key]
            string = string + char
            #char = pytesseract.image_to_string(PIL_frame, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789.-') # whitelist characters
            #print(w, h, np.mean(frame[y:y+h, x:x+w]))
            #cv2.imshow("bounding boxes", frame[y:y+h, x:x+w].astype(np.uint8))
            #cv2.waitKey(50)
        #string = float(string.replace(" ","").replace(",",".").split("(Oe)")[0])
        #print(string)
        string = float(string)
    except:
        cv2.imshow("ocr", frame.astype(np.uint8))
        cv2.waitKey(100)
        string = float(input("could not read by ocr: input field by hand: "))
    return string
