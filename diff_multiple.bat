rem ドラッグアンドドロップしたファイルのパスで実行。
cd %~dp0
:loop
if not "%~nx1"=="" (
  start diff.bat %~dp1%~nx1 & shift & goto loop
)