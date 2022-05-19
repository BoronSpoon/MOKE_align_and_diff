cd %~dp0
python diff_no_ocr_changing_basis_frame.py 1 %*
rem 第一引数は差分画像の白黒の協調具合、第二引数は動画のパス(ドラッグアンドドロップ)
rem cmd /k 
rem コメントアウトを外した場合ウィンドウを閉じない