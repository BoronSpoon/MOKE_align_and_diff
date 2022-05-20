from ctypes import alignment
import cv2
import numpy as np 
from ocr import *
from plot import *
import sys
import math
import multiprocessing as mp
from matplotlib.widgets import SpanSelector

def open_video(path):
    """動画を読み込み

    Args:
        path (str): 動画のパス

    Returns:
        frames: 動画の各画像を格納するリスト
    """    
    cap = cv2.VideoCapture(path)
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    frames = []
    # 動画の最後のフレームまで読む
    while(cap.isOpened()):
        # 1フレームずつ読み込む
        ret, frame = cap.read()
        if ret == True:
            frames.append(frame)
        else:
            break
    cap.release()
    return frames

def split_frames(frames, fields):
    """動画を基準画像と測定画像に分ける

    Args:
        frames (list of numpy 2D array): 動画の各画像を格納するリスト
        fields (list of float): 動画の各画像の磁界強度を格納するリスト

    Returns:
        basis_frame (numpy 2D array): 基準画像
        frames (list of numpy 2D array): 基準画像を除いた動画の各画像(測定画像)を格納するリスト
        original_frames (list of numpy 2D array):  動画の各画像を格納するリスト
        fields (list of float): 動画の各画像の磁界強度を格納するリスト
    """
    original_frames = frames # 元のframesを退避
    frames = [crop_frame(frame) for frame in frames] # 緑色の文字が端にあるので除く
    basis_frame = frames[0] # 基準画像
    frames = frames[1:] # 基準画像以外の画像
    fields = fields[1:] # framesの枚数に合わせる
    return basis_frame, frames, original_frames, fields

def crop_frame(frame):
    """画像の端にある緑色の文字をクロップする

    Args:
        frame (list of numpy 2D array): 動画の各画像を格納するリスト

    Returns:
        frame (list of numpy 2D array): クロップした動画の各画像を格納するリスト
    """
    h, w = frame.shape[:2]
    frame = frame[int(h*1/6):int(h*5/6),: , :] # 緑色の文字が端にあるので除く
    return frame

def shift2csv(shifts, shift_csv_path, frames_offset_count):
    """各画像のずれをCSV形式で出力

    Args:
        shifts (list of list of float): 各画像のずれを格納するリスト
        shift_csv_path (str): 出力するCSVのパス
        frames_offset_count (int): 基準画像にする画像の番号
    """
    with open(shift_csv_path, "w", newline ="") as f:  
        writer = csv.writer(f)
        header = [f"frame count \n(start at {frames_offset_count + 2}'th frame)", "round(shift_x)", "round(shift_y)", "shift_x", "shift_y"]
        writer.writerow(header) # write header
        for count, shift in enumerate(shifts): # write row by row
            writer.writerow([count, shift[0], shift[1], shift[2], shift[3]])

def align_frame_map(args):
    """マルチプロセス処理に必要なマップ関数

    Args:
        args (list): 引数のリスト

    Returns:
        align_frame(*args) (function with args): 引数付きの関数
    """
    return align_frame(*args)

def align_frames_multiprocessing(basis_frame, frames, basis_frame_shift=(0,0,0,0), alignment_mode="orb", shift_mode="normal", shift_flag=True):
    """マルチプロセス処理による測定画像のアライメント

    Args:
        basis_frame (numpy 2D array): 基準画像
        frame (list of numpy 2D array): 測定画像を格納するリスト
        basis_frame_shift (tuple, optional): 「各グループの基準画像」の「最初の基準画像」を起点としたずれ. Defaults to (0,0,0,0).
        alignment_mode (str, optional): アライメントに用いる手法(orb,ecc). Defaults to "orb".
        shift_mode (str, optional): ずれ補正に用いる手法(normal,integer_shift). Defaults to "normal".

    Returns:
        aligned_frames (list of numpy 2D array): アライメントを行った測定画像
        shift (list of list of float and int): 基準画像からのずれの大きさ
    """
    aligned_frames = []
    shifts = []
    num_processes = mp.cpu_count()
    p = mp.Pool(num_processes)
    args = [[basis_frame, frame, basis_frame_shift, alignment_mode, shift_mode, shift_flag] for frame in frames]
    rets = p.map(align_frame_map, args)
    for ret in rets:
        aligned_frames.append(ret[0])
        shifts.append(ret[1])
    return aligned_frames, shifts

def align_frames(basis_frame, frames, basis_frame_shift=(0,0,0,0), alignment_mode="orb", shift_mode="normal", shift_flag=True):
    """測定画像のアライメント

    Args:
        basis_frame (numpy 2D array): 基準画像
        frame (list of numpy 2D array): 測定画像を格納するリスト
        basis_frame_shift (tuple, optional): 「各グループの基準画像」の「最初の基準画像」を起点としたずれ. Defaults to (0,0,0,0).
        alignment_mode (str, optional): アライメントに用いる手法(orb,ecc). Defaults to "orb".
        shift_mode (str, optional): ずれ補正に用いる手法(normal,integer_shift). Defaults to "normal".

    Returns:
        aligned_frames (list of numpy 2D array): アライメントを行った測定画像
        shift (list of list of float and int): 基準画像からのずれの大きさ
    """
    aligned_frames = []
    shifts = []
    for frame in frames:
        if shift_flag == True:
            aligned_frame, shift = align_frame(basis_frame, frame, basis_frame_shift, alignment_mode, shift_mode)
            aligned_frames.append(aligned_frame)
            shifts.append(shift)
        elif shift_flag == False:
            aligned_frames.append(frame)
            shifts.append([0,0,0,0])
    return aligned_frames, shifts

def align_frame(frame1, frame2, basis_frame_shift=(0,0,0,0), alignment_mode="normal", shift_mode="normal"):
    """2枚の画像のアライメント

    Args:
        frame1 (numpy 2D array): 1枚目の画像
        frame2 (numpy 2D array): 2枚目の画像
        basis_frame_shift (tuple, optional): 「各グループの基準画像」の「最初の基準画像」を起点としたずれ. Defaults to (0,0,0,0).
        alignment_mode (str, optional): アライメントに用いる手法(orb,ecc). Defaults to "orb".
        shift_mode (str, optional): ずれ補正に用いる手法(normal,integer_shift). Defaults to "normal".

    Returns:
        frame2_aligned (numpy 2D array): アライメントを行った画像
        shift (list of float and int): 基準画像からのずれの大きさ
    """
    # Convert images to grayscale
    frame1_gray = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    if alignment_mode == "ecc":
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
    elif alignment_mode == "orb":
        # Define the motion model
        orb = cv2.ORB_create(5000)
        queryKeypoints, queryDescriptors = orb.detectAndCompute(frame1_gray, None)
        trainKeypoints, trainDescriptors = orb.detectAndCompute(frame2_gray, None)
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = list(matcher.match(queryDescriptors,trainDescriptors, None))   
        matches.sort(key=lambda x: x.distance, reverse=False)
        matches = matches[:int(len(matches) * 0.10)]
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)
        for count, match in enumerate(matches):
            points1[count, :] = queryKeypoints[match.queryIdx].pt
            points2[count, :] = trainKeypoints[match.trainIdx].pt
        warp_matrix, mask = cv2.estimateAffinePartial2D(points1, points2)

    # add relative shift
    warp_matrix[0,2] += basis_frame_shift[2]
    warp_matrix[1,2] += basis_frame_shift[3]

    # get shift (x,y)
    shift = [round(warp_matrix[0,2]), round(warp_matrix[1,2]), warp_matrix[0,2], warp_matrix[1,2]]

    # change shift to integer
    if shift_mode == "normal":
        warp_matrix = np.array([
            [1, 0, warp_matrix[0,2]],
            [0, 1, warp_matrix[1,2]],
        ]).astype(np.float32)
    elif shift_mode == "integer_shift":
        warp_matrix = np.array([
            [1, 0, round(warp_matrix[0,2])],
            [0, 1, round(warp_matrix[1,2])],
        ]).astype(np.float32)

    # Use warpAffine for Translation, Euclidean and Affine
    frame2_aligned = cv2.warpAffine(frame2, warp_matrix, (frame1.shape[1], frame1.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return frame2_aligned, shift

def write_frames_to_avi(frames, path):
    """各画像をAVI動画ファイルに出力

    Args:
        frame (list of numpy 2D array): 画像を格納するリスト
        path (str): 動画の保存先のパス
    """
    writer = cv2.VideoWriter(path,  
                            cv2.VideoWriter_fourcc(*'MJPG'), 
                            10, basis_frame.shape[:2][::-1])
    for frame in frames:
        writer.write(frame)
    writer.release()

def get_max_diff(basis_frame, frames):
    """差分画像の最大値を取得

    Args:
        basis_frame (numpy 2D array): 基準画像
        frame (list of numpy 2D array): 測定画像を格納するリスト

    Returns:
        max_diff (float): 差分画像の最大値
    """
    diffs = []
    h, w = basis_frame.shape[:2]
    basis_frame = basis_frame[20:h-20, 20:w-20, :] # avoid edge black color due to alignment
    for frame in frames:
        frame = frame[20:h-20, 20:w-20, :] # avoid edge black color due to alignment
        basis_frame = basis_frame.astype("float32") # uint8で -1=255 になるのを防ぐ
        frame = frame.astype("float32")
        diff = np.max(np.abs(frame/2 - basis_frame/2)) # 差分画像の最大値
        diffs.append(diff)
    return max(diffs)

def get_diff_frames(basis_frame, frames, max_diff, rate):
    """差分画像を取得

    Args:
        basis_frame (numpy 2D array): 基準画像
        frame (list of numpy 2D array): 測定画像を格納するリスト
        max_diff (float): 差分画像の最大値
        rate (int): コントラスト（白黒）の増加具合

    Returns:
        diff_frames (list of numpy 2D array): 差分画像
    """
    status = True
    print("press wasd to change 'rate'")
    print("press Enter, Esc to exit")
    while (status):
        diff_frames = []
        len_frames = len(frames)
        for count, frame in enumerate(frames):
            diff_frame = get_diff_frame(basis_frame, frame, max_diff, rate)        
            diff_frames.append(diff_frame)        
            cv2.imshow(path, diff_frame)
            if count == len_frames -1: # last frame: wait infinitely for "left", "right", "enter"
                if status != False: # break when status = False
                    k = cv2.waitKey(0) 
            else:
                k = cv2.waitKey(10) 
            if k in [13, 27]: # "enter" or "esc" key to break
                status = False # break out of loop
                if count == len_frames -1: 
                    break # if broke at middle, diff_frames will not be fully polulated
            if status != False: # prevent breaking when status = False
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

    return diff_frames

def get_diff_frame(frame1, frame2, max_diff, rate):
    """2枚の画像の差分画像を取得

    Args:
        frame1 (numpy 2D array): 1枚目の画像
        frame2 (numpy 2D array): 2枚目の画像
        max_diff (float): 差分画像の最大値
        rate (int): コントラスト（白黒）の増加具合

    Returns:
        diff_frame (numpy 2D array): 差分画像
    """
    frame1 = frame1.astype("float32")
    frame2 = frame2.astype("float32")
    diff_frame = (frame2/2 - frame1/2) * 127*rate/max_diff + 127 # -127~127 + 127(offset) => 0~255(縁を除く)
    # 255以上と0以下の値をそれぞれ255,0にする。overflowした値は255を法としてしまうため。
    diff_frame = np.where(diff_frame>255, 255, diff_frame)
    diff_frame = np.where(diff_frame<0, 0, diff_frame)
    diff_frame = diff_frame.astype("uint8") # uint8に戻す
    return diff_frame

def select_hysterisis_region(xmin, xmax):
    global hystersis_region_list
    hystersis_region_list.append([xmin, xmax])

def correct_hysterisis(fields_, contrasts_, hystersis_region_list):
    fields = np.array(fields_)
    contrasts = np.array(contrasts_)
    if len(hystersis_region_list) == 0:
        slope = 0
    else:
        slopes = []
        for hysterisis_region in hystersis_region_list:
            cropped_fields = fields[np.where((hysterisis_region[0]<fields)|(fields<hysterisis_region[1]))]
            cropped_contrasts = contrasts[np.where((hysterisis_region[0]<fields)|(fields<hysterisis_region[1]))]
            slopes.append(np.polyfit(cropped_fields,cropped_contrasts,1)[0])
        slope = sum(slopes)/len(slopes)
    corrected_contrasts = contrasts - fields*slope
    return list(corrected_contrasts)

def select_region_and_get_contrast(basis_frame, frames, fields, path, plot_path, corrected_plot_path, contrast_csv_path):
    """コントラストを得る領域を選択

    Args:
        basis_frame (numpy 2D array): 基準画像
        frame (list of numpy 2D array): 測定画像を格納するリスト
        fields (list of float): 測定画像の磁界強度を格納するリスト
        path (str): ウィンドウの名前(ファイルの保存先と同じ)
        plot_path (str): プロットの保存先
        contrast_csv_path (str): コントラストのCSVの保存先
    """
    status = True
    get_coords_setup(basis_frame, path) # コントラスト測定範囲の設定をするための事前の準備(loopしてほしくないもの)
    fig, axes = plt.subplots(1, 2, figsize=(9,5), tight_layout=True) # 2つ(axes[0],axes[1])のsubplotを作る
    span = SpanSelector(axes[0], select_hysterisis_region, 'horizontal', useblit=True, rectprops=dict(alpha=0.5, facecolor='red')) # axes[0]の領域を選択できるようにする
    while(status):
        status = get_coords(basis_frame, path) # コントラスト測定範囲の設定
        contrasts = get_contrast(frames, path) # コントラストの測定
        corrected_contrasts = correct_hysterisis(fields, contrasts, hystersis_region_list) # コントラストの傾き補正
        plot_contrast(fig, axes, fields, contrasts, corrected_contrasts) # コントラスト対磁界のプロット
    if contrast_csv_path is not None: # 文字認識が行われている場合
        save_contrast(fields, contrasts, corrected_contrasts, plot_path, corrected_plot_path) # コントラスト対磁界のファイル保存
        contrast2csv(fields, contrasts, contrast_csv_path) # コントラスト対磁界のcsv出力

if __name__ == "__main__":
    rate = float(sys.argv[1]) # 差分画像の白黒の強調具合
    path = sys.argv[2] # 動画のパス
    alignment_mode = sys.argv[3] # orb or ecc (orb is faster and accurate)
    shift_mode = sys.argv[4] # normal or integer_shift
    ocr_flag = sys.argv[5] == "True" # 文字認識を使うか？磁界強度の記入されていない動画ではFalseにする
    shift_flag = sys.argv[6] == "True" # 位置合わせを使うか？位置合わせの必要ない場合はFalseにする
    # CONFIG
    #alignment_mode="orb" # orb or ecc (orb is faster and accurate)
    #shift_mode="normal" # normal or integer_shift
    #ocr_flag=True # 文字認識を使うか？磁界強度の記入されていない動画ではFalseにする
    
    if ocr_flag == True:
        plot_path = path.replace(".avi","_contrast.png") # contrast plot path
        corrected_plot_path = path.replace(".avi","_corrected_contrast.png") # contrast plot path
        contrast_csv_path = path.replace(".avi","_contrast.csv") # contrast csv path
    else: # 文字認識を行わない場合
        plot_path = None
        corrected_plot_path = path.replace(".avi","_corrected_contrast.png") # contrast plot path
        contrast_csv_path = None
    diff_path = path.replace(".avi","_diff.avi") # diff avi path
    meas_path = path.replace(".avi","_meas.avi") # meas avi path
    shift_csv_path = path.replace(".avi","_shift.csv") # shift csv path(基準画像と測定画像のシフト量)
    frames_offset_count = 0 # 基準画像にする画像の番号

    frames = open_video(path) # 動画の読み込み
    if ocr_flag == True:
        fields = get_strings(frames) # 磁場の強さをocrで取得
    else: # 文字認識を行わない場合
        fields = [0.0 for i in range(len(frames))] # 仮のデータを提供
    # 基準画像とその他に分割
    basis_frame, frames, original_frames, fields = split_frames(frames, fields)
    first_basis_frame = basis_frame # 最初の基準画像を退避
    aligned_frames = []
    shifts = []
    hystersis_region_list = [] # ヒステリシスの補正に使う領域のリスト
    basis_frame_shift = [0, 0, 0, 0] # 「各グループのbasis frame」の「一番最初のbasis frame」から見たshift
    group_frame_count = len(frames) # number of total frames (dont split frames)
    #group_frame_count = 10 # number of frames in group
    group_num = int(len(frames)/group_frame_count) # number of groups
    for group in range(group_num):
        print(f"group {group+1}/{group_num}")
        # グループに属するフレーム
        group_frames = frames[group*group_frame_count: (group+1)*group_frame_count]
        # 基準画像とその他に分割
        group_aligned_frames, group_shifts = align_frames(basis_frame, group_frames, 
            basis_frame_shift=basis_frame_shift, alignment_mode=alignment_mode, shift_mode=shift_mode, shift_flag=shift_flag) # 基準・測定画像の位置合わせ
        aligned_frames.extend(group_aligned_frames)
        shifts.extend(group_shifts)
        basis_frame = group_frames[-1] # 最後のフレームが次のグループのbasis frame
        basis_frame_shift = group_shifts[-1]
    basis_frame = first_basis_frame # 基準画像を最初の基準画像に戻す
    shift2csv(shifts, shift_csv_path, frames_offset_count) # アライメントの際のオフセット(x,y)をcsvに出力
    max_diff = get_max_diff(basis_frame, aligned_frames) # 差分画像の差分の最大値を得る
    diff_frames = get_diff_frames(basis_frame, aligned_frames, max_diff, rate) # 画像の差分を取る
    print("saving video to file")
    write_frames_to_avi(aligned_frames, meas_path)
    write_frames_to_avi(diff_frames, diff_path)
    if ocr_flag == True:
        select_region_and_get_contrast(basis_frame, frames, fields, path, plot_path, corrected_plot_path, contrast_csv_path) # contrast on diff_frame -> meas_frame

    #.destroyAllWindows() 
    print("The video was successfully saved") 
