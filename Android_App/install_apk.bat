@echo off
REM Quick script to install APK on connected Android device

echo.
echo ==========================================
echo    Face ID App - Quick Installer
echo ==========================================
echo.

REM Check for ADB
where adb >nul 2>nul
if errorlevel 1 (
    echo ERROR: ADB not found in PATH!
    echo.
    echo Please install Android SDK Platform Tools:
    echo https://developer.android.com/studio/releases/platform-tools
    echo.
    pause
    exit /b 1
)

REM Check for connected devices
echo Checking for connected devices...
adb devices
echo.

REM Find APK
set DEBUG_APK=app\build\outputs\apk\debug\app-debug.apk

if not exist "%DEBUG_APK%" (
    echo ERROR: APK not found!
    echo Please build the APK first by running: build_apk.bat
    echo.
    pause
    exit /b 1
)

echo Installing APK...
echo.
adb install -r "%DEBUG_APK%"

if errorlevel 1 (
    echo.
    echo Installation failed!
    echo.
    echo Common issues:
    echo   - USB debugging not enabled
    echo   - Device not authorized
    echo   - App already installed (try: adb uninstall com.faceid)
    echo.
) else (
    echo.
    echo ==========================================
    echo    Installation Successful!
    echo ==========================================
    echo.
    echo The Face ID app is now installed on your device.
    echo You can find it in your app drawer.
    echo.
)

pause
