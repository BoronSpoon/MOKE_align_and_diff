rem ドラッグアンドドロップしたファイルのパスで実行。
cd %~dp0
:loop
if not "%~nx1"=="" (
  start diff_no_ocr_changing_basis_frame.bat %~dp1%~nx1 & shift & goto loop
)