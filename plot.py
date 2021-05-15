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
    status = 1
    while True: 
        cv2.imshow(path,frame)
        if C.done:
            C.done = False
            break
        elif cv2.waitKey(1) in [13,27]: # "enter" or "esc" key to break
            status = 0
            break
    return status

def get_contrast(frames, path, contrast_types):
    contrasts_dict = {}
    for contrast_type in contrast_types:
        contrasts_dict[contrast_type] = []
    for frame in frames:
        #print(C.plist)
        x = [min(C.plist[0][1], C.plist[1][1]), max(C.plist[0][1], C.plist[1][1])]
        y = [min(C.plist[0][0], C.plist[1][0]), max(C.plist[0][0], C.plist[1][0])]
        cropped_frame = frame[x[0]:x[1], y[0]:y[1], :] # コントラストを測る部分にクロップ
        cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY) # コントラストは白黒画像で測定
        cropped_frame = cropped_frame.astype("float64")
        #cv2.imshow(plot, cropped_frame)
        #cv2.waitKey(100)
        contrasts = {}
        contrasts["RMS_contrast"] = cropped_frame.std()
        contrasts["mean_contrast"] = 100 * (cropped_frame.max()-cropped_frame.min())/cropped_frame.mean()
        contrasts["mean_intensity"] = cropped_frame.mean()
        for contrast_type in contrast_types:
            contrasts_dict[contrast_type].append(contrasts[contrast_type])
    return contrasts_dict
    
def plot_contrast(fields, contrasts_dict, plot_path_dict):
    for key in contrasts_dict.keys():
        contrasts = contrasts_dict[key]
        plot_path = plot_path_dict[key]
        plt.clf()
        fields = [field for field, contrast in zip(fields, contrasts) if contrast != None]
        contrasts = [contrast for contrast in contrasts if contrast != None]
        plt.plot(fields, contrasts)
        plt.xlabel("Magnetic field intensity (Oe)")
        plt.ylabel(key)
        plt.savefig(plot_path)
        plt.ion()
        plt.show() # do not block (continue the program even when plot windows is not closed)
        plt.pause(.001)

def contrast2csv(fields, contrasts_dict, contrast_csv_path_dict):
    for key in contrasts_dict.keys():
        contrasts = contrasts_dict[key]
        contrast_csv_path = contrast_csv_path_dict[key]
        with open(contrast_csv_path, "w", newline ="") as f:  
            writer = csv.writer(f)
            header = ["field strength (Oe)", "contrast"]
            writer.writerow(header) # write header
            for field, contrast in zip(fields,contrasts): # write row by row
                writer.writerow([field, contrast])

