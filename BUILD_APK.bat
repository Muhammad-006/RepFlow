@echo off
setlocal
title RepFlow - build Android APK
cd /d "%~dp0"

set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY (
    echo Python was not found. Install Python 3.10+ from https://www.python.org/downloads/
    pause & exit /b 1
)

%PY% tools\write_build_config.py

cd client
if not exist ".venv\.installed" (
    echo [RepFlow] Installing the Flet build tools ^(one time^)...
    %PY% -m venv .venv || goto :fail
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt "flet-cli>=0.86.0,<0.87.0" || goto :fail
    echo ok> ".venv\.installed"
)

echo.
echo [RepFlow] Building the APK. The FIRST build downloads Flutter + the Android SDK
echo           automatically and can take 10-20 minutes. Later builds take ~2-4 minutes.
echo.
".venv\Scripts\flet.exe" build apk --yes || goto :fail

echo.
echo  Done! Your APK is in:  client\build\apk\
echo  Copy it to the phone (USB / Google Drive / WhatsApp) and install it.
explorer "build\apk"
pause
exit /b 0

:fail
echo.
echo  Build failed - see the messages above. Tip: run it again; the first build
echo  sometimes times out while downloading the Android SDK.
pause
exit /b 1
