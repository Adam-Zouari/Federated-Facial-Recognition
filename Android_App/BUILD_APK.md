# 📦 Building APK Guide

## Quick Build APK

### Method 1: Using Android Studio (Easiest)

1. **Open Android Studio** with your project
2. **Build Menu** → **Build Bundle(s) / APK(s)** → **Build APK(s)**
3. Wait for build to complete
4. Click **"locate"** link in notification
5. APK is at: `app/build/outputs/apk/debug/app-debug.apk`

**Or for Release APK:**
- **Build Menu** → **Build Bundle(s) / APK(s)** → **Build APK(s)**
- APK at: `app/build/outputs/apk/release/app-release-unsigned.apk`

### Method 2: Using Command Line (Windows)

**Debug APK:**
```cmd
cd Android_App
gradlew.bat assembleDebug
```
Output: `app\build\outputs\apk\debug\app-debug.apk`

**Release APK:**
```cmd
cd Android_App
gradlew.bat assembleRelease
```
Output: `app\build\outputs\apk\release\app-release-unsigned.apk`

---

## APK Types Explained

### Debug APK
- ✅ Quick to build
- ✅ Easy to install
- ✅ Good for testing
- ❌ Larger size
- ❌ Not optimized
- **Use for:** Development, testing on your devices

### Release APK (Unsigned)
- ✅ Optimized and smaller
- ✅ Better performance
- ❌ Needs signing for Play Store
- **Use for:** Personal distribution, sideloading

### Release APK (Signed)
- ✅ Can be published to Play Store
- ✅ Users can install it
- ✅ Production ready
- **Use for:** Public distribution

---

## Building Signed Release APK

### Step 1: Generate Signing Key

**Using Android Studio:**
1. **Build** → **Generate Signed Bundle / APK**
2. Select **APK**
3. Click **Create new...**
4. Fill in:
   - Key store path: Choose location (e.g., `faceid-release.jks`)
   - Password: Your password
   - Alias: `faceid`
   - Validity: 25 years
   - Certificate info: Your details
5. Click **OK**

**Using Command Line:**
```cmd
keytool -genkey -v -keystore faceid-release.jks -alias faceid -keyalg RSA -keysize 2048 -validity 10000
```

### Step 2: Configure Signing in Project

Create `keystore.properties` in project root:
```properties
storePassword=YOUR_KEYSTORE_PASSWORD
keyPassword=YOUR_KEY_PASSWORD
keyAlias=faceid
storeFile=../faceid-release.jks
```

**IMPORTANT:** Add to `.gitignore`:
```
keystore.properties
*.jks
*.keystore
```

### Step 3: Update `app/build.gradle.kts`

The file is already configured! Just make sure your `keystore.properties` exists.

### Step 4: Build Signed APK

**Using Android Studio:**
1. **Build** → **Generate Signed Bundle / APK**
2. Select **APK**
3. Select your keystore
4. Enter passwords
5. Choose **release** build variant
6. Click **Finish**

**Using Command Line:**
```cmd
gradlew.bat assembleRelease
```

Output: `app\build\outputs\apk\release\app-release.apk` (signed!)

---

## Complete Build Script

I'll create a batch file to automate everything:

**File:** `build_apk.bat`

```batch
@echo off
echo ==========================================
echo    Face ID App - APK Builder
echo ==========================================
echo.

echo [1/3] Cleaning previous builds...
call gradlew.bat clean

echo.
echo [2/3] Building Debug APK...
call gradlew.bat assembleDebug

echo.
echo [3/3] Building Release APK...
call gradlew.bat assembleRelease

echo.
echo ==========================================
echo    Build Complete!
echo ==========================================
echo.
echo Debug APK:
echo   app\build\outputs\apk\debug\app-debug.apk
echo.
echo Release APK:
echo   app\build\outputs\apk\release\app-release-unsigned.apk
echo.
echo To install on device:
echo   adb install app\build\outputs\apk\debug\app-debug.apk
echo.
pause
```

---

## Installation Methods

### Method 1: Direct Install via USB

```cmd
adb install app\build\outputs\apk\debug\app-debug.apk
```

### Method 2: Copy to Device

1. Copy APK to phone
2. Open file on phone
3. Tap "Install"
4. Enable "Install from Unknown Sources" if prompted

### Method 3: Share via Cloud

1. Upload APK to Google Drive / Dropbox
2. Share link
3. Download on Android device
4. Install

---

## Troubleshooting

### "gradlew.bat not found"
Make sure you're in the `Android_App` directory:
```cmd
cd Android_App
```

### "Java not found"
Install Java Development Kit (JDK):
1. Download JDK 17 from Oracle or OpenJDK
2. Set JAVA_HOME environment variable

### "Build failed"
1. Ensure model.pt exists in assets
2. Run: `gradlew.bat clean`
3. Try again

### "Cannot install APK"
1. Enable "Unknown Sources" in Android settings
2. Check if app is already installed (uninstall first)
3. Ensure Android version is 7.0+

### APK too large
- Use ProGuard (already enabled in release)
- Compress assets
- Remove unused resources

---

## APK Size Optimization

Already configured in `app/build.gradle.kts`:

```kotlin
buildTypes {
    release {
        isMinifyEnabled = true        // ✓ Removes unused code
        isShrinkResources = true      // ✓ Removes unused resources
        proguardFiles(...)            // ✓ Obfuscates code
    }
}
```

**Expected sizes:**
- Debug APK: ~70-90 MB
- Release APK (unsigned): ~50-70 MB
- Release APK (signed): ~50-70 MB

---

## Publishing to Google Play Store

### Requirements:
1. ✅ Signed Release APK or AAB (Android App Bundle)
2. ✅ Google Play Developer Account ($25 one-time)
3. ✅ App icon, screenshots, description
4. ✅ Privacy policy

### Steps:
1. Build signed AAB:
   ```cmd
   gradlew.bat bundleRelease
   ```
   Output: `app/build/outputs/bundle/release/app-release.aab`

2. Go to Google Play Console
3. Create new app
4. Upload AAB
5. Fill in app details
6. Submit for review

---

## Quick Reference

| Task | Command |
|------|---------|
| **Debug APK** | `gradlew.bat assembleDebug` |
| **Release APK** | `gradlew.bat assembleRelease` |
| **Clean Build** | `gradlew.bat clean` |
| **Install APK** | `adb install path/to/app.apk` |
| **List Devices** | `adb devices` |
| **Uninstall** | `adb uninstall com.faceid` |

---

## Distribution Options

### 1. Direct Distribution
- Share APK file directly
- Users install manually
- No approval process needed

### 2. Google Play Store
- Official app store
- Automatic updates
- Requires approval (~3 days)

### 3. Alternative Stores
- Amazon App Store
- Samsung Galaxy Store
- F-Droid (open source)

### 4. Enterprise Distribution
- Internal company apps
- MDM solutions
- Corporate app stores

---

## Next Steps

1. ✅ Run `build_apk.bat` (I'll create this)
2. ✅ Find APK in `app/build/outputs/apk/`
3. ✅ Install on your Android device
4. ✅ Test the app
5. 📱 Share with others!

Happy building! 🎉
