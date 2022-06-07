@echo OFF
set script_dir=%~dp0
set runs_dir=%script_dir%runs
set "reply=y"
set /p "reply=Clean runs? [y|n]: "
if /i not "%reply%" == "y" goto :eof
del /q %runs_dir%\events.out.tfevents*