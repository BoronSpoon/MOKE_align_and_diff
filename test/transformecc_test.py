import numpy as np
import cv2
import os
cwd = os.path.dirname(__file__)

def align_frame(frame1, frame2):
    # Convert images to grayscale
    frame1_gray = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    # Find size of frame1
    sz = frame1.shape

    # Define the motion model
    warp_mode = cv2.MOTION_TRANSLATION # only xy_shift
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    number_of_iterations = 5000

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC (frame1_gray,frame2_gray,warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=1)

    # get shift (x,y)
    shift = [round(warp_matrix[0,2]), round(warp_matrix[1,2]), warp_matrix[0,2], warp_matrix[1,2]]

    # Use warpAffine for Translation, Euclidean and Affine
    frame2_aligned = cv2.warpAffine(frame2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return frame2_aligned, shift

cap = cv2.VideoCapture(os.path.join(cwd, "test2.avi"))
_, frame1 = cap.read() # read first frame
[cap.read() for i in range(290)]
_, frame2 = cap.read() # read 22th frame
frame1 = frame1[100:-100, 100:-100]
frame2 = frame2[100:-100, 100:-100]
frame3, shift = align_frame(frame1, frame2)
#diff_frame = frame1-frame3
diff_frame = np.zeros((frame3.shape[0], frame3.shape[1], 3), dtype="uint8")
#diff_frame[:,:,0] = frame1[:,:,0] # B
#diff_frame[:,:,2] = frame2[:,:,0] # R
#cv2.imshow("a", ((diff_frame-np.min(diff_frame))*1))
#cv2.imshow("a", diff_frame*1)
#cv2.waitKey(0)

for i in range(1000000):
    cv2.imshow("press 'q' to escape", frame1[20:-20,20:-20,0])
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    cv2.imshow("press 'q' to escape", frame3[20:-20,20:-20,0])
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
