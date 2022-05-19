## required packages
#### pip install
numpy, matplotlib, pytesseract, Pillow, opencv-contrib-python
#### System install
tesseract
#### Path
directory of tesseract.exe (usually C:\Program Files\Tesseract-OCR)

## 使い方
1. Frameloopでmeas（測定）画像を選んだ状態で動画に出力する。出力された動画をdiff.batにドラッグアンドドロップする。(diff_multiple.batは複数動画を同時に処理する用)
2. 処理されて画像が表示されるまで待つ。
3. 画像の表示されている画面をクリックして、”wasdで白黒の協調具合を調整する。調整後はEscかEnterで終了する。
4. コントラストを得る領域を選択する。何回でも領域の設定をしなおすことができる。領域の設定後はEscかEnterで終了する。
5. ファイルが出力される。
   1. _diff.avi 差分画像
   2. _meas.avi アライメントをした測定画像
   3. _shift.csv アライメントの際に用いた各フレームのx,y方向のシフト
   4. _contrast.csv 各フレームのコントラスト
   5. _contrast.png コントラストのプロット

## How to use
1. drop measured image .avi created using Frameloop onto diff.bat (drop multiple files on diff_multiple.bat)
2. wait until image pops up.
3. Click on the screen and use "wasd" to control contrast. Press Esc or Enter to finish.
4. Select region to get contrast. Reselect region until satisfied. Press Esc or Enter to finish.
5. Files are output
   1. _diff.avi differential image
   2. _meas.avi aligned measured image
   3. _shift.csv output shift in x,y direction used for alignment
   4. _contrast.csv output contrasts for each frame
   5. _contrast.png output contrast plot

## references
tesseract install guide: https://medium.com/quantrium-tech/installing-and-using-tesseract-4-on-windows-10-4f7930313f82