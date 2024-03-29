import cv2
import numpy as np 
from ocr import *
from plot import *
import sys
import math
from statistics import mean
import multiprocessing as mp
import os

contrast_types = [
    "RMS_contrast",
    "mean_contrast",
    "mean_intensity"
    ]

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
    if len(frames) == 0:
        print("Error path file is empty")
        return None
    return frames

def remove_first_frames(n, frames, fields):
    return frames[n:], fields[n:]

def split_frames(frames, fields):
    frames = [crop_frame(frame) for frame in frames] # 緑色の文字が端にあるので除く
    basis_frame = frames[0] # 基準画像
    frames = frames[1:] # 基準画像以外の画像
    fields = fields[1:] # framesの枚数に合わせる
    return basis_frame, frames, fields

def crop_frame(frame):
    h, w = frame.shape[:2]
    frame = frame[int(h*1/6):int(h*5/6),: , :] # 緑色の文字が端にあるので除く
    return frame

def shifts2csv(shifts, shifts_csv_path, frames_offset_count):
    with open(shifts_csv_path, "w", newline ="") as f:  
        writer = csv.writer(f)
        header = [f"frame count \n(start at {frames_offset_count + 2}'th frame)", "round(shift_x)", "round(shift_y)", "shift_x", "shift_y"]
        writer.writerow(header) # write header
        for count, shift in enumerate(shifts): # write row by row
            writer.writerow([count, shift[0], shift[1], shift[2], shift[3]])

def csv2shifts(shifts_csv_path):
    with open(shifts_csv_path, "r", newline ="") as f:  
        reader = csv.reader(f, delimiter=",")
        shifts = []
        reader = [i for i in reader][1:] # eliminate header
        for count, shift0, shift1, shift2, shift3 in reader: # read row by row consisting of (count, shift0~3)
            shift0 = float(shift0)
            shift1 = float(shift1)
            shift2 = float(shift2)
            shift3 = float(shift3)
            shifts.append([shift0, shift1, shift2, shift3])
        return shifts

def fields2csv(fields, fields_csv_path, frames_offset_count):
    with open(fields_csv_path, "w", newline ="") as f:
        writer = csv.writer(f)
        header = [f"frame count \n(start at {frames_offset_count + 2}'th frame)", "field (Oe)"]
        writer.writerow(header) # write header
        for count, field in enumerate(fields): # write row by row
            writer.writerow([count, field])

def csv2fields(fields_csv_path):
    with open(fields_csv_path, "r", newline ="") as f:
        reader = csv.reader(f, delimiter=",")
        fields = []
        reader = [i for i in reader][1:] # eliminate header
        for count, field in reader: # read row by row consisting of (count, field)
            field = float(field)
            fields.append(field)
        return fields

def align_frame_map(args):
    return align_frame(*args)

def align_frames_multiprocessing(basis_frame, frames, meas_path):
    aligned_frames = []
    shifts = []
    writer = cv2.VideoWriter(meas_path,  
                            cv2.VideoWriter_fourcc(*'RGBA'), # no compression
                            10, basis_frame.shape[:2][::-1])
    writer.write(basis_frame) # write basis_frame
    num_processes = mp.cpu_count()
    p = mp.Pool(num_processes)
    args = [[basis_frame, frame] for frame in frames]
    rets = p.map(align_frame_map, args)
    for ret in rets:
        aligned_frames.append(ret[0])
        writer.write(ret[0])
        shifts.append(ret[1])
    writer.release()
    return aligned_frames, shifts

def align_frames(basis_frame, frames, meas_path):
    aligned_frames = []
    shifts = []
    writer = cv2.VideoWriter(meas_path,  
                            cv2.VideoWriter_fourcc(*'RGBA'), # no compression
                            10, basis_frame.shape[:2][::-1])
    writer.write(basis_frame) # write basis_frame
    for frame in frames:
        aligned_frame, shift = align_frame(basis_frame, frame)
        aligned_frames.append(aligned_frame)
        shifts.append(shift)
        writer.write(aligned_frame)
    writer.release()
    return aligned_frames, shifts

def align_frame(frame1, frame2):
    # Convert images to grayscale
    frame1_gray = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    # Find size of frame1
    sz = frame1.shape

    # Define the motion model
    warp_mode = cv2.MOTION_TRANSLATION # only xy_shift
    #warp_mode = cv2.MOTION_EUCLIDEAN # only xy_shift and rotation
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    number_of_iterations = 5000

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    try:
        (cc, warp_matrix) = cv2.findTransformECC (frame1_gray, frame2_gray, warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=1)
    except:
        (cc, warp_matrix) = cv2.findTransformECC (frame1_gray, frame2_gray, warp_matrix, warp_mode, criteria)

    # get shift (x,y)
    shift = [round(warp_matrix[0,2]), round(warp_matrix[1,2]), warp_matrix[0,2], warp_matrix[1,2]]

    # Use warpAffine for Translation, Euclidean and Affine
    frame2_aligned = cv2.warpAffine(frame2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return frame2_aligned, shift

def get_mean_min_max_diff(basis_frame, frames):
    mean_diffs = []
    min_diffs = []
    max_diffs = []
    h, w = basis_frame.shape[:2]
    basis_frame = basis_frame[20:h-20, 20:w-20, :] # avoid edge black color due to alignment
    #basis_frame_max = np.max(basis_frame)
    for frame in frames:
        frame = frame[20:h-20, 20:w-20, :] # avoid edge black color due to alignment
        basis_frame = basis_frame.astype("float32") # uint8で -1=255 になるのを防ぐ
        frame = frame.astype("float32")
        #frame = frame/(basis_frame/basis_frame_max)
        mean_diff = np.mean(frame - basis_frame) # 差分画像のmean
        min_diff = np.min(frame - basis_frame) # 差分画像のmin
        max_diff = np.max(frame - basis_frame) # 差分画像のmax
        mean_diffs.append(mean_diff)
        min_diffs.append(min_diff)
        max_diffs.append(max_diff)
    return mean(mean_diffs), min(min_diffs), max(max_diffs)

def get_hist(fig, line1, diff_frame):
    hist = cv2.calcHist([diff_frame], [0], None, [256], [0, 255])
    hist /= hist.sum() # normalization 
    plt.ylim(0,hist.max()) 
    line1.set_ydata(hist.ravel()) # update data
    fig.canvas.draw() # redraw the canvas)
    hist_frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8) # convert canvas to image
    hist_frame = hist_frame.reshape(fig.canvas.get_width_height()[::-1] + (3,)) # resize
    hist_frame = cv2.cvtColor(hist_frame, cv2.COLOR_RGB2BGR) # img is rgb, convert to opencv's default bgr
    return hist_frame

def get_diff_frames(basis_frame, frames, mean_diff, min_diff, max_diff, rate, diff_path, hist_path):
    print(mean_diff, min_diff, max_diff)
    diff_writer = cv2.VideoWriter(diff_path,  
                            cv2.VideoWriter_fourcc(*'MJPG'), 
                            10, basis_frame.shape[:2][::-1])

    # prepare hist figures
    fig = plt.figure()
    x = np.arange(0,256)
    y = np.zeros(256)
    line1, = plt.plot(x, y)
    plt.title("Grayscale Histogram (Normalized)")
    plt.xlabel("Bins")
    plt.ylabel("% of Pixels")

    hist = cv2.calcHist([basis_frame], [0], None, [256], [0, 256])
    line1.set_ydata(hist.ravel()) # update data
    fig.canvas.draw() # redraw the canvas                   
    hist_shape = fig.canvas.get_width_height()
    hist_writer = cv2.VideoWriter(hist_path,  
                            cv2.VideoWriter_fourcc(*'MJPG'), 
                            10, hist_shape)
    status = 1
    print("press wasd to change 'rate'")
    print("press Enter, Esc to exit")
    while (status):
        diff_frames = []
        len_frames = len(frames)
        basis_frame_max = np.max(basis_frame)
        for count, frame in enumerate(frames):
            diff_frame = get_diff_frame(basis_frame, frame, basis_frame_max, mean_diff, min_diff, max_diff, rate)        
            diff_frames.append(diff_frame)
            cv2.imshow(path, diff_frame)
            if count == len_frames -1: # last frame: wait infinitely for "left", "right", "enter"
                if status != 0: # break when status = 0
                    k = cv2.waitKey(0) 
            else:
                k = cv2.waitKey(10) 
            if k in [13, 27]: # "enter" or "esc" key to break
                status = 0 # break out of loop
                if count == len_frames -1: 
                    break # if broke at middle, diff_frames will not be fully polulated
            if status != 0: # prevent breaking when status = 0
                if k == ord("w"): # "w" to increment "rate"
                    rate += 5
                    break
                elif k == ord("d"): # "d" to increment "rate"
                    rate += 1
                    break
                elif k == ord("s"): # "a","s" to decrement "rate"
                    rate -= 5
                    break
                elif k == ord("a"): # "a","s" to decrement "rate"
                    rate -= 1
                    break

        print(f"rate={int(rate)}")
            
    for diff_frame in diff_frames:
        hist_frame = get_hist(fig, line1, diff_frame) # plot hist and write to avi
        diff_writer.write(diff_frame)
        hist_writer.write(hist_frame)
    diff_writer.release() 
    hist_writer.release() 
    return diff_frames

def get_diff_frame(basis_frame, meas_frame, basis_frame_max, mean_diff, min_diff, max_diff, rate):
    basis_frame = basis_frame.astype("float32") # reference
    meas_frame = meas_frame.astype("float32")
    #meas_frame = meas_frame/(basis_frame/basis_frame_max)
    max_magnitude = max(abs(min_diff - mean_diff), abs(max_diff - mean_diff))
    diff_frame = (meas_frame - basis_frame) - mean_diff # mean -> 0
    #diff_frame = diff_frame/basis_frame
    #diff_frame = diff_frame * (127*rate/max_magnitude) + 127 # -127~127 + 127
    diff_frame = diff_frame*rate + 127
    # 255以上と0以下の値をそれぞれ255,0にする。overflowした値は255を法としてしまうため。
    diff_frame = np.where(diff_frame>255, 255, diff_frame)
    diff_frame = np.where(diff_frame<0, 0, diff_frame)
    diff_frame = diff_frame.astype("uint8") # uint8に戻す
    return diff_frame

def select_region_and_get_contrast(diff_frames, fields, path, plot_path_dict, contrast_csv_path_dict):
    plt.clf()
    status = 1
    get_coords_setup(diff_frames[0], path) # コントラスト測定範囲の設定をするための事前の準備(loopしてほしくないもの)
    while(status):
        status = get_coords(diff_frames[0], path) # コントラスト測定範囲の設定
        contrasts = get_contrast(diff_frames, path, contrast_types) # コントラストの測定
        plot_contrast(fields, contrasts, plot_path_dict) # コントラスト対磁界のプロット
    contrast2csv(fields, contrasts, contrast_csv_path_dict) # コントラスト対磁界のcsv出力

if __name__ == "__main__":
    rate = float(sys.argv[1]) # 差分画像の白黒の強調具合
    path = sys.argv[2] # 動画のパス
    diff_path = path.replace(".avi","_diff.avi") # diff avi path
    meas_path = path.replace(".avi","_meas.avi") # meas avi path
    hist_path = path.replace(".avi","_hist.avi") # hist avi path
    plot_path_dict = {}
    contrast_csv_path_dict = {}
    for contrast_type in contrast_types:
        plot_path_dict[contrast_type] = path.replace(".avi",f"_{contrast_type}.png") # contrast plot path
        contrast_csv_path_dict[contrast_type] = path.replace(".avi",f"_{contrast_type}.csv") # contrast csv path
    fields_csv_path = path.replace(".avi","_fields.csv") # fields path (磁場データ)
    shifts_csv_path = path.replace(".avi","_shifts.csv") # shifts csv path (基準画像と測定画像のシフト量)
    frames_offset_count = 1 # 基準画像にする画像のフレーム
    frames = open_video(path) # 動画の読み込み 
    # アライメントを以前に行った場合、その結果を用いる
    processed_before = False
    if (os.path.isfile(meas_path) and os.path.isfile(fields_csv_path) and os.path.isfile(shifts_csv_path)):
        aligned_frames = open_video(meas_path) # アライメント済み動画の読み込み
        if aligned_frames is not None: # meas_frameが空でない場合
            fields = csv2fields(fields_csv_path) # 磁場データの読み込み
            shifts = csv2shifts(shifts_csv_path) # シフトデータの読み込み
            result = cv2.matchTemplate(frames[frames_offset_count], aligned_frames[0], cv2.TM_CCOEFF_NORMED) # 前回の画像と今回の画像の比較
            if np.any(result>0.99): # 以前処理された画像と今回処理する画像が同じであるか
                processed_before = True
                print("this video has been processed before. Reusing processed data")
                basis_frame, frames = aligned_frames[0], aligned_frames[1:]
    # アライメントを以前に行っていない場合
    if not processed_before:
        fields = get_strings_multiprocessing(frames) # 磁場の強さをocrで取得
        frames, fields = remove_first_frames(frames_offset_count, frames, fields) # 最初の画像は基準画像なのでH=0の画像ではないようにする。
        # 基準画像とその他に分割
        basis_frame, frames, fields = split_frames(frames, fields)
        frames, shifts = align_frames_multiprocessing(basis_frame, frames, meas_path) # 基準・測定画像の位置合わせ
        #basis_frame = np.average(np.array(frames), axis=0)# 平均画像を基準画像とする
        shifts2csv(shifts, shifts_csv_path, frames_offset_count) # アライメントの際のオフセット(x,y)をcsvに出力
        fields2csv(fields, fields_csv_path, frames_offset_count) # 磁場データをcsvに出力
    mean_diff, min_diff, max_diff = get_mean_min_max_diff(basis_frame, frames) # 差分画像の差分のmean,min,maxを得る
    diff_frames = get_diff_frames(basis_frame, frames, mean_diff, min_diff, max_diff, rate, diff_path, hist_path) # 画像の差分を取る
    select_region_and_get_contrast(diff_frames, fields, path, plot_path_dict, contrast_csv_path_dict)

    #.destroyAllWindows() 
    print("The video was successfully saved") 
