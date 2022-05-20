import matplotlib.pyplot as plt
import csv
import numpy as np
import cv2
import config as C

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

def get_coords_setup(frame, path): # コントラスト測定範囲の設定をするための事前の準備(loopしてほしくないもの)
    print("select region to measure contrast")
    print("press Enter, Esc to exit")
    cv2.namedWindow(path)
    original_frame = np.copy(frame)
    cv2.setMouseCallback(path, onMouse, [frame, original_frame])

def get_coords(frame, path): # コントラスト測定範囲の設定
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

def get_contrast(frames, path):
    contrasts = []
    for frame in frames:
        #print(C.plist)
        x = [min(C.plist[0][1], C.plist[1][1]), max(C.plist[0][1], C.plist[1][1])]
        y = [min(C.plist[0][0], C.plist[1][0]), max(C.plist[0][0], C.plist[1][0])]
        cropped_frame = frame[x[0]:x[1], y[0]:y[1], :] # コントラストを測る部分にクロップ
        cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY) # コントラストは白黒画像で測定
        cropped_frame = cropped_frame.astype("float64")
        #cv2.imshow(plot, cropped_frame)
        #cv2.waitKey(100)
        # RMS contrast
        RMS_contrast = cropped_frame.std()
        # mean contrast
        mean_contrast = 100 * (cropped_frame.max()-cropped_frame.min())/cropped_frame.mean()
        # mean intensity
        mean_pixel = cropped_frame.mean()
        #contrasts.append(mean_contrast) # use RMS constant
        contrasts.append(mean_pixel) # use pixel intensity
    return contrasts
    
def save_contrast(fig, axes, fields, contrasts, corrected_contrasts, plot_path, corrected_plot_path):
    plt.clf()
    plt.plot(fields, contrasts)
    plt.xlabel("Magnetic field intensity (Oe)")
    plt.ylabel("Contrast")
    plt.savefig(plot_path)
    plt.clf()
    plt.plot(fields, corrected_contrasts)
    plt.xlabel("Magnetic field intensity (Oe)")
    plt.ylabel("Corrected contrast")
    plt.savefig(corrected_plot_path)
    
def plot_contrast(fig, axes, fields, contrasts, corrected_contrasts):
    axes[0].cla()
    axes[1].cla()
    axes[0].plot(fields, contrasts)
    axes[0].set_xlabel("Magnetic field intensity (Oe)")
    axes[0].set_ylabel("Contrast")
    axes[1].plot(fields, corrected_contrasts)
    axes[1].set_xlabel("Magnetic field intensity (Oe)")
    axes[1].set_ylabel("Contrast")
    plt.ion()
    plt.show() # do not block (continue the program even when plot windows is not closed)
    plt.pause(.001)

def contrast2csv(fields, contrasts, contrast_csv_path):
    with open(contrast_csv_path, "w", newline ="") as f:  
        writer = csv.writer(f)
        header = ["field strength (Oe)", "contrast"]
        writer.writerow(header) # write header
        for field, contrast in zip(fields,contrasts): # write row by row
            writer.writerow([field, contrast])

