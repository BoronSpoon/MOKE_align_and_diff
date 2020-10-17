import cv2
import numpy as np 
from ocr import get_strings
from plot import *
import sys

# todo 
# shift csv
# 

def open_video(path):
    cap = cv2.VideoCapture(path)
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    frames = []
    fields = []
    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            frames.append(frame)
            fields.append(get_strings(frame))
        else:
            break
    cap.release()
    return frames, fields

def remove_first_frames(n, frames, fields):
    return frames[n:], fields[n:]

def split_frames(frames, fields):
    original_frames = frames # 元のframesを退避
    frames = [crop_frame(frame) for frame in frames] # 緑色の文字が端にあるので除く
    basis_frame = frames[0] # 基準画像
    frames = frames[1:] # 基準画像以外の画像
    fields = fields[1:] # framesの枚数に合わせる
    return basis_frame, frames, original_frames, fields

def crop_frame(frame):
    h, w = frame.shape[:2]
    frame = frame[int(h*1/6):int(h*5/6),: , :] # 緑色の文字が端にあるので除く
    return frame

def align_frames(basis_frame, frames, meas_path):
    aligned_frames = []
    diff_frames = []
    writer = cv2.VideoWriter(meas_path,  
                            cv2.VideoWriter_fourcc(*'MJPG'), 
                            10, basis_frame.shape[:2][::-1]) 
    for frame in frames:
        aligned_frame = align_frame(basis_frame, frame)
        aligned_frames.append(aligned_frame)
        writer.write(aligned_frame)
    writer.release()
    return aligned_frames

def align_frame(frame1, frame2):
    # Convert images to grayscale
    frame1_gray = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    # Find size of frame1
    sz = frame1.shape

    # Define the motion model
    #warp_mode = cv2.MOTION_EUCLEDIAN # xy_shift + rotation
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

    # Use warpAffine for Translation, Euclidean and Affine
    frame2_aligned = cv2.warpAffine(frame2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return frame2_aligned

def get_max_diff(basis_frame, frames):
    diffs = []
    h, w = basis_frame.shape[:2]
    basis_frame = basis_frame[20:h-20, 20:w-20, :] # avoid edge black color due to alignment
    for frame in frames:
        frame = frame[20:h-20, 20:w-20, :] # avoid edge black color due to alignment
        basis_frame = basis_frame.astype("float32") # uint8で -1=255 になるのを防ぐ
        frame = frame.astype("float32")
        diff = np.max(np.abs(frame/2 - basis_frame/2)) # 差分画像の最大値
        diffs.append(diff)
    print(max(diffs))
    return max(diffs)

def get_diff_frames(basis_frame, frames, max_diff, rate, diff_path):
    diff_frames = []
    writer = cv2.VideoWriter(diff_path,  
                            cv2.VideoWriter_fourcc(*'MJPG'), 
                            10, basis_frame.shape[:2][::-1]) 
    for frame in frames:
        diff_frame = get_diff_frame(basis_frame, frame, max_diff, rate)        
        diff_frames.append(diff_frame)
        writer.write(diff_frame)
    writer.release() 
    return diff_frames

def get_diff_frame(frame1, frame2, max_diff, rate):
    frame1 = frame1.astype("float32")
    frame2 = frame2.astype("float32")
    diff_frame = (frame2/2 - frame1/2) * 127*rate/max_diff + 127 # -127~127 + 127(offset) => 0~255(縁を除く)
    # 255以上と0以下の値をそれぞれ255,0にする。overflowした値は255を法としてしまうため。
    diff_frame = np.where(diff_frame>255, 255, diff_frame)
    diff_frame = np.where(diff_frame<0, 0, diff_frame)
    diff_frame = diff_frame.astype("uint8") # uint8に戻す
    cv2.imshow(path, diff_frame)
    cv2.waitKey(1)
    return diff_frame

rate = float(sys.argv[1]) # 差分画像の白黒の強調具合
path = sys.argv[2] # 動画のパス
diff_path = path.replace(".avi","_diff.avi") # diff avi path
meas_path = path.replace(".avi","_meas.avi") # meas avi path
plot_path = path.replace(".avi","_contrast.png") # contrast plot path
contrast_csv_path = path.replace(".avi","_contrast.csv") # contrast csv path
shift_csv_path = path.replace(".avi","_shift.csv") # shift csv path(基準画像と測定画像のシフト量)

frames, fields = open_video(path) # 動画の読み込み 
frames, fields = remove_first_frames(0, frames, fields)# 最初の画像は基準画像なのでH=0の画像ではないようにする。
# 基準画像とその他に分割
basis_frame, frames, original_frames, fields = split_frames(frames, fields)
frames = align_frames(basis_frame, frames, meas_path) # 基準・測定画像の位置合わせ
max_diff = get_max_diff(basis_frame, frames) # 差分画像の差分の最大値を得る
frames = get_diff_frames(basis_frame, frames, max_diff, rate, diff_path) # 画像の差分を取る

get_coords(basis_frame, path) # コントラスト測定範囲の設定
contrasts = get_contrast(frames, path) # コントラストの測定
plot_contrast(fields, contrasts, plot_path) # コントラスト対磁界のプロット
contrast2csv(fields, contrasts, contrast_csv_path) # コントラスト対磁界のcsv出力

#.destroyAllWindows() 
print("The video was successfully saved") 
