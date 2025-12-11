# ✅ APK Build Checklist

## Prerequisites

- [ ] Android Studio installed (or JDK 17+ for command line)
- [ ] Model converted: `python convert_model_to_mobile.py`
- [ ] File exists: `Android_App/app/src/main/assets/model.pt`

## Building APK

### Option 1: Automated Build (Easiest!) ⭐

```cmd
cd Android_App
build_apk.bat
```

This will:
- ✅ Clean previous builds
- ✅ Build debug APK
- ✅ Build release APK
- ✅ Show you where APKs are located
- ✅ Offer to open the folder

### Option 2: Android Studio

1. Open Android Studio
2. Open `Android_App` folder
3. **Build** → **Build Bundle(s) / APK(s)** → **Build APK(s)**
4. Click "locate" in the notification
5. Done! ✅

### Option 3: Command Line

**Debug APK:**
```cmd
cd Android_App
gradlew.bat assembleDebug
```

**Release APK:**
```cmd
cd Android_App
gradlew.bat assembleRelease
```

## Installing APK

### Option 1: Automated Install ⭐

```cmd
cd Android_App
install_apk.bat
```

### Option 2: Manual Install via USB

1. Enable USB debugging on your Android device
2. Connect device to computer
3. Run:
```cmd
adb install app\build\outputs\apk\debug\app-debug.apk
```

### Option 3: Copy to Phone

1. Copy APK from `app\build\outputs\apk\debug\app-debug.apk`
2. Transfer to your phone (USB, email, cloud, etc.)
3. Open the APK file on your phone
4. Tap "Install"
5. Enable "Unknown Sources" if prompted

## APK Locations

After building, find your APKs here:

**Debug APK:**
```
Android_App/app/build/outputs/apk/debug/app-debug.apk
```

**Release APK (unsigned):**
```
Android_App/app/build/outputs/apk/release/app-release-unsigned.apk
```

**Release APK (signed):**
```
Android_App/app/build/outputs/apk/release/app-release.apk
```
(Only if you configured signing)

## Creating Signed Release APK (Optional)

For distribution or Play Store:

1. Generate keystore:
```cmd
keytool -genkey -v -keystore faceid-release.jks -alias faceid -keyalg RSA -keysize 2048 -validity 10000
```

2. Create `Android_App/keystore.properties`:
```properties
storePassword=YOUR_PASSWORD
keyPassword=YOUR_PASSWORD
keyAlias=faceid
storeFile=faceid-release.jks
```

3. Build signed APK:
```cmd
gradlew.bat assembleRelease
```

## Troubleshooting

### ❌ "gradlew.bat not found"
**Solution:** Make sure you're in the `Android_App` directory:
```cmd
cd Android_App
```

### ❌ "Java not found"
**Solution:** Install JDK 17 or newer:
- Download: https://adoptium.net/
- Set JAVA_HOME environment variable

### ❌ "Build failed"
**Solution:**
1. Check if `model.pt` exists in `app/src/main/assets/`
2. Run: `gradlew.bat clean`
3. Try building again

### ❌ "Cannot install APK on phone"
**Solution:**
1. Enable "Unknown Sources" or "Install Unknown Apps" in Settings
2. Check if app is already installed (uninstall first)
3. Verify Android version is 7.0 or higher

### ❌ APK won't open folder
**Solution:** Manually navigate to:
```
Android_App\app\build\outputs\apk\
```

## Quick Reference

| What you want | Command |
|---------------|---------|
| **Build everything** | `build_apk.bat` |
| **Install on device** | `install_apk.bat` |
| **Debug APK only** | `gradlew.bat assembleDebug` |
| **Release APK only** | `gradlew.bat assembleRelease` |
| **Clean build** | `gradlew.bat clean` |
| **Check devices** | `adb devices` |
| **Uninstall app** | `adb uninstall com.faceid` |

## What's Next?

After building APK:

1. ✅ Install on your Android device
2. ✅ Grant camera permission
3. ✅ Test face enrollment
4. ✅ Test face verification
5. 📱 Share APK with others!

## File Sizes

Approximate APK sizes:

- **Debug APK**: 70-90 MB
  - Larger because includes debug symbols
  - Good for testing

- **Release APK**: 50-70 MB
  - Optimized and compressed
  - Good for distribution

The model file (`model.pt`) is the largest component (~30-50 MB).

---

**That's it!** You can now build and install your Face ID Android app. 🎉

For more details, see `BUILD_APK.md`
