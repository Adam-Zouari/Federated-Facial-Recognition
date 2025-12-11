@echo off
REM ==========================================
REM    Face ID App - APK Builder
REM ==========================================

echo.
echo ==========================================
echo    Face ID App - APK Builder
echo ==========================================
echo.

REM Check if we're in the right directory
if not exist "gradlew.bat" (
    echo ERROR: gradlew.bat not found!
    echo Please run this script from the Android_App directory.
    echo.
    pause
    exit /b 1
)

REM Check if model exists
if not exist "app\src\main\assets\model.pt" (
    echo WARNING: model.pt not found in assets!
    echo Please run: python convert_model_to_mobile.py
    echo.
    set /p continue="Continue anyway? (y/n): "
    if /i not "%continue%"=="y" exit /b 1
)

echo.
echo [1/4] Cleaning previous builds...
call gradlew.bat clean
if errorlevel 1 (
    echo ERROR: Clean failed!
    pause
    exit /b 1
)

echo.
echo [2/4] Building Debug APK...
call gradlew.bat assembleDebug
if errorlevel 1 (
    echo ERROR: Debug build failed!
    pause
    exit /b 1
)

echo.
echo [3/4] Building Release APK...
call gradlew.bat assembleRelease
if errorlevel 1 (
    echo ERROR: Release build failed!
    pause
    exit /b 1
)

echo.
echo [4/4] Checking output...

REM Check if APKs were created
set DEBUG_APK=app\build\outputs\apk\debug\app-debug.apk
set RELEASE_APK=app\build\outputs\apk\release\app-release-unsigned.apk

echo.
echo ==========================================
echo    Build Complete!
echo ==========================================
echo.

if exist "%DEBUG_APK%" (
    echo [DEBUG APK]
    echo   Location: %DEBUG_APK%
    for %%I in ("%DEBUG_APK%") do echo   Size: %%~zI bytes
    echo.
) else (
    echo WARNING: Debug APK not found!
    echo.
)

if exist "%RELEASE_APK%" (
    echo [RELEASE APK]
    echo   Location: %RELEASE_APK%
    for %%I in ("%RELEASE_APK%") do echo   Size: %%~zI bytes
    echo.
) else (
    echo WARNING: Release APK not found!
    echo.
)

echo ==========================================
echo    Installation Instructions
echo ==========================================
echo.
echo Option 1: Install via USB (ADB)
echo   adb install %DEBUG_APK%
echo.
echo Option 2: Copy to phone
echo   1. Copy APK file to your phone
echo   2. Open the file on your phone
echo   3. Tap "Install"
echo.
echo Option 3: Open folder
echo   start app\build\outputs\apk\debug
echo.

set /p open="Open APK folder? (y/n): "
if /i "%open%"=="y" (
    start "" "app\build\outputs\apk"
)

echo.
pause
