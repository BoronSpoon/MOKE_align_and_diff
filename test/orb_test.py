import numpy as np
import cv2
import os
cwd = os.path.dirname(__file__)

def align_frame_orb(frame1, frame2):
    # Convert images to grayscale
    frame1_gray = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    # Define the motion model
    orb = cv2.ORB_create(5000)
    queryKeypoints, queryDescriptors = orb.detectAndCompute(frame1_gray,None)
    trainKeypoints, trainDescriptors = orb.detectAndCompute(frame2_gray,None)
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = list(matcher.match(queryDescriptors,trainDescriptors, None))   
    matches.sort(key=lambda x: x.distance, reverse=False)
    matches = matches[:int(len(matches) * 0.10)]
    cv2.imwrite(os.path.join(cwd,"matches.jpg"), cv2.drawMatches(frame1, queryKeypoints, frame2, trainKeypoints, matches, None))
 
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for count, match in enumerate(matches):
        points1[count, :] = queryKeypoints[match.queryIdx].pt
        points2[count, :] = trainKeypoints[match.trainIdx].pt

    #warp_matrix, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
    warp_matrix, mask = cv2.estimateAffinePartial2D(points1, points2)

    # get shift (x,y)
    shift = [round(warp_matrix[0,2]), round(warp_matrix[1,2]), warp_matrix[0,2], warp_matrix[1,2]]

    # Use warpAffine for Translation, Euclidean and Affine
    frame2_aligned = cv2.warpAffine(frame2, warp_matrix, (frame1.shape[1], frame1.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    #frame2_aligned = cv2.warpPerspective(frame2, warp_matrix, (frame1.shape[1], frame1.shape[0]))

    return frame2_aligned, shift

cap = cv2.VideoCapture(os.path.join(cwd, "test2.avi"))
_, frame1 = cap.read() # read first frame
[cap.read() for i in range(290)]
_, frame2 = cap.read() # read 22th frame
frame1 = frame1[30:, 100:-100]
frame2 = frame2[30:, 100:-100]
frame3, shift = align_frame_orb(frame1, frame2)
#diff_frame = frame1-frame3
diff_frame = np.zeros((frame3.shape[0], frame3.shape[1], 3), dtype="uint8")
#diff_frame[:,:,0] = frame1[:,:,0] # B
#diff_frame[:,:,2] = frame2[:,:,0] # R
#cv2.imshow("a", ((diff_frame-np.min(diff_frame))*1))
#cv2.imshow("a", diff_frame*1)
#cv2.waitKey(0)

for i in range(1000000):
    cv2.imshow("press 'q' to escape", frame1[100:-100,20:-20,0])
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    cv2.imshow("press 'q' to escape", frame3[100:-100,20:-20,0])
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
