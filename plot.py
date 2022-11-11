import matplotlib.pyplot as plt
import csv
import numpy as np
import cv2
import config as C
import matplotlib

def onMouse(event, x, y, flag, params):
    img1, img2 = params
    if event == cv2.EVENT_LBUTTONDOWN: # 矩形の1点目を指定
        C.drawing = True
        C.plist[0] = x,y

    elif event == cv2.EVENT_MOUSEMOVE: # マウスを動かすときは矩形を描画
        if C.drawing == True:
            cv2.addWeighted(img2, 1.0, img1, 0.0, 0.0, img1) # 元画像を描画(前回の矩形を消すため)
            cv2.rectangle(img1,(C.plist[0][0],C.plist[0][1]),(x,y),(0,0,255),1) # 矩形の描画

    elif event == cv2.EVENT_LBUTTONUP: # 矩形の終了点を指定
        C.plist[1] = x,y
        cv2.addWeighted(img2, 1.0, img1, 0.0, 0.0, img1) # 元画像を描画(前回の矩形を消すため)
        cv2.rectangle(img1,(C.plist[0][0],C.plist[0][1]),(C.plist[1][0],C.plist[1][1]),(0,0,255),1)
        if C.plist[0][0] == C.plist[1][0] and C.plist[0][1] == C.plist[1][1]: # 領域の面積が0の時
            C.drawing = False
            C.done = False
        else:
            C.drawing = False
            C.done = True

def get_coords_setup(frame, path): # 輝度平均測定範囲の設定をするための事前の準備(loopしてほしくないもの)
    print("select region to measure average_intensity")
    print("press Enter, Esc to exit")
    cv2.namedWindow(path)
    original_frame = np.copy(frame)
    cv2.setMouseCallback(path, onMouse, [frame, original_frame])

def get_coords(frame, path): # 輝度平均測定範囲の設定
    status = True
    while True: 
        cv2.imshow(path,frame)
        if C.done:
            C.done = False
            break
        elif cv2.waitKey(1) in [13,27]: # "enter" or "esc" key to break
            status = False
            break
    return status

def get_average_intensity(frames, path):
    average_intensities = []
    for frame in frames:
        #print(C.plist)
        x = [min(C.plist[0][1], C.plist[1][1]), max(C.plist[0][1], C.plist[1][1])]
        y = [min(C.plist[0][0], C.plist[1][0]), max(C.plist[0][0], C.plist[1][0])]
        cropped_frame = frame[x[0]:x[1], y[0]:y[1], :] # 輝度平均を測る部分にクロップ
        cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY) # 輝度平均は白黒画像で測定
        cropped_frame = cropped_frame.astype("float64")
        #cv2.imshow(plot, cropped_frame)
        #cv2.waitKey(100)
        # RMS average_intensity
        RMS_average_intensity = cropped_frame.std()
        # mean average_intensity
        mean_average_intensity = 100 * (cropped_frame.max()-cropped_frame.min())/cropped_frame.mean()
        # mean intensity
        mean_pixel = cropped_frame.mean()
        #average_intensities.append(mean_average_intensity) # use RMS constant
        average_intensities.append(mean_pixel) # use pixel intensity
    return average_intensities
    
def save_average_intensity(fields, average_intensities, corrected_average_intensities, plot_path, corrected_plot_path, ocr_flag):
    if ocr_flag: # 文字認識が行われている場合、ヒステリシスをプロット。
        xlabel = "Magnetic Field Intensity (Oe)"
    else: # MOVIEモードの場合、輝度平均の時間応答をプロット
        xlabel = "Frame Count"
    plt.clf()
    plt.plot(fields, average_intensities, ocr_flag)
    if not ocr_flag: # y軸をスケール
        min_ = np.min(corrected_average_intensities)
        max_ = np.max(corrected_average_intensities)
        diff_ = max_ - min_
        plt.ylim([min_-diff_*0.1, max_+diff_*0.1])
    plt.xlabel(xlabel)
    plt.ylabel("Average Intensity")
    plt.savefig(plot_path)
    plt.clf()
    plt.plot(fields, corrected_average_intensities, ocr_flag)
    if not ocr_flag: # y軸をスケール
        min_ = np.min(corrected_average_intensities)
        max_ = np.max(corrected_average_intensities)
        diff_ = max_ - min_
        plt.ylim([min_-diff_*0.1, max_+diff_*0.1])
    plt.xlabel(xlabel)
    plt.ylabel("Corrected Average Intensity")
    plt.savefig(corrected_plot_path)
    
def plot_average_intensity(fig, ax, fields, average_intensities, corrected_average_intensities, hysterisis_region_list, ocr_flag):
    if ocr_flag: # 文字認識が行われている場合、ヒステリシスをプロット。
        xlabel = "Magnetic Field Intensity (Oe)"
    else: # MOVIEモードの場合、輝度平均の時間応答をプロット
        xlabel = "Frame Count"
    ax.cla()
    ax.plot(fields, average_intensities)
    ax.plot(fields, corrected_average_intensities)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Average Intensity")
    y_min, y_max = ax.get_ylim()
    for hysterisis_region in hysterisis_region_list:
        ax.add_patch(matplotlib.patches.Rectangle((hysterisis_region[0], y_min), hysterisis_region[1]-hysterisis_region[0], y_max-y_min, color="red", alpha=0.2))
    plt.show() # do not block (continue the program even when plot windows is not closed)
    plt.pause(.001)

def average_intensity2csv(fields, average_intensities, average_intensity_csv_path, ocr_flag):
    if ocr_flag: # 文字認識が行われている場合、ヒステリシスをプロット。
        xlabel = "Magnetic Field Intensity (Oe)"
    else: # MOVIEモードの場合、輝度平均の時間応答をプロット
        xlabel = "Frame Count"
    with open(average_intensity_csv_path, "w", newline ="") as f:  
        writer = csv.writer(f)
        header = [xlabel, "Average Intensity"]
        writer.writerow(header) # write header
        for field, average_intensity in zip(fields,average_intensities): # write row by row
            writer.writerow([field, average_intensity])

