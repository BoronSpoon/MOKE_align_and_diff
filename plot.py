import matplotlib.pyplot as plt
import numpy as np

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
        C.drawing = False
        C.done = True

def get_coords(frame, path): # コントラスト測定範囲の設定
    cv2.namedWindow(path)
    original_frame = np.copy(frame)
    cv2.setMouseCallback(path, onMouse, [frame, original_frame])
    while True: 
        cv2.imshow(path,frame)
        if cv2.waitKey(1) & 0xFF == 27 or C.done == True:
            break
    #cv2.destroyAllWindows()

def get_contrast(frames, path):
    contrasts = []
    for frame in frames:
        #print(C.plist)
        x = [min(C.plist[0][1], C.plist[1][1]), max(C.plist[0][1], C.plist[1][1])]
        y = [min(C.plist[0][0], C.plist[1][0]), max(C.plist[0][0], C.plist[1][0])]
        cropped_frame = frame[x[0]:x[1], y[0]:y[1], :] # コントラストを測る部分にクロップ
        cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY) # コントラストは白黒画像で測定
        #cv2.imshow(plot, cropped_frame)
        #cv2.waitKey(100)
        # RMS contrast
        RMS_contrast = cropped_frame.std()
        contrasts.append(RMS_contrast) # use RMS constant
    return contrasts
    
def plot_contrast(x,y,plot_path):
    plt.plot(x,y)
    plt.xlabel("Magnetic field intensity (Oe)")
    plt.ylabel("Contrast")
    plt.savefig(plot_path)
    plt.show()
