@echo off
setlocal
title RepFlow AI Coach Server
cd /d "%~dp0server"

set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY goto :nopython

if not exist ".venv\.installed" (
    echo.
    echo  [RepFlow] First run - setting up the AI server ^(one time, ~1-2 minutes^)...
    echo.
    %PY% -m venv .venv || goto :fail
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :fail
    echo ok> ".venv\.installed"
)

".venv\Scripts\python.exe" server.py %*
pause
exit /b 0

:nopython
echo.
echo  Python was not found. Install Python 3.10+ from https://www.python.org/downloads/
echo  (tick "Add python.exe to PATH" during install), then run this file again.
pause
exit /b 1

:fail
echo.
echo  Setup failed - check your internet connection and run START_SERVER.bat again.
if exist ".venv\.installed" del ".venv\.installed"
pause
exit /b 1
