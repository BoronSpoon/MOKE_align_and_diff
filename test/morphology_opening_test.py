import cv2
import os
#cwd = os.path.dirname(__file__)
cwd = os.path.abspath('') # for ipython notebook
cap = cv2.VideoCapture(os.path.join(cwd, "test.avi"))
_, frame = cap.read() # read first frame
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
frame_morph = 255-cv2.morphologyEx(255-frame, cv2.MORPH_OPEN, kernel) # invert color for open morphology
cv2.imshow("frames", cv2.hconcat([frame, frame_morph]))
while(True):  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break