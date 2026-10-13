@echo off
REM Lets phones on your Wi-Fi / hotspot reach the RepFlow server (Windows Firewall).
REM Run once. It will ask for administrator permission.
net session >nul 2>&1
if %errorlevel% neq 0 (
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)
netsh advfirewall firewall delete rule name="RepFlow AI Coach" >nul 2>&1
netsh advfirewall firewall add rule name="RepFlow AI Coach" dir=in action=allow protocol=TCP localport=8765 profile=any
netsh advfirewall firewall add rule name="RepFlow AI Coach" dir=in action=allow protocol=UDP localport=8766-8767 profile=any
echo.
echo  Firewall rules added: TCP 8765 (AI server) and UDP 8766-8767 (auto-discovery).
pause
