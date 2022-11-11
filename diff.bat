cd %~dp0
python diff.py 1 %* orb normal False True
rem 第一引数は差分画像の白黒の協調具合、第二引数は動画のパス(ドラッグアンドドロップ)
cmd /k 
rem コメントアウトを外した場合ウィンドウを閉じない