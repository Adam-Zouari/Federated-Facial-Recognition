# ✅ Android App Setup Checklist

Use this checklist to ensure everything is set up correctly!

## 🔧 Prerequisites

- [ ] Python environment with PyTorch installed
- [ ] Trained model exists at `checkpoints/local/vggface2_weak/best_model.pth` (or another path)
- [ ] Android Studio installed (download from https://developer.android.com/studio)
- [ ] Android device with USB debugging enabled OR Android emulator (API 24+)

---

## 📝 Step-by-Step Setup

### Phase 1: Model Conversion

- [ ] Open terminal/command prompt in project root directory
- [ ] Run: `python convert_model_to_mobile.py`
- [ ] Verify success message appears
- [ ] Check file exists: `Android_App/app/src/main/assets/model.pt`
- [ ] Note the model size (should be 30-50 MB)

**Troubleshooting:**
- If script fails, check Python can import torch
- If model not found, specify path: `python convert_model_to_mobile.py --model path/to/your/model.pth`

---

### Phase 2: Android Studio Setup

- [ ] Launch Android Studio
- [ ] Click "Open" (not "New Project")
- [ ] Navigate to and select `Android_App` folder
- [ ] Click "OK"
- [ ] Wait for Gradle sync to complete (check bottom status bar)
- [ ] Verify no errors in "Build" tab at bottom

**Troubleshooting:**
- If Gradle sync fails: File → Invalidate Caches → Invalidate and Restart
- If missing SDK: Tools → SDK Manager → Install Android 7.0+ (API 24)

---

### Phase 3: Device/Emulator Setup

**Option A: Physical Device**
- [ ] Enable Developer Options on device
  - Settings → About Phone → Tap "Build Number" 7 times
- [ ] Enable USB Debugging
  - Settings → Developer Options → USB Debugging → ON
- [ ] Connect device via USB cable
- [ ] Verify device appears in Android Studio (top toolbar)

**Option B: Emulator**
- [ ] Click "Device Manager" (phone icon) in Android Studio
- [ ] Click "Create Device"
- [ ] Select a phone (e.g., Pixel 5)
- [ ] Download system image (API 24 or higher)
- [ ] Click "Finish"
- [ ] Start emulator (▶ play button)

---

### Phase 4: Build and Run

- [ ] Select device/emulator in top toolbar
- [ ] Click green "Run" button (▶) OR press Shift+F10
- [ ] Wait for build to complete
- [ ] App launches on device/emulator
- [ ] Grant camera permission when prompted

**Troubleshooting:**
- Build fails: Clean project (Build → Clean Project) then rebuild
- App crashes: Check Logcat (bottom panel) for errors
- Camera not working: Check permissions in device settings

---

### Phase 5: Test the App

- [ ] **Test 1: Add Face**
  - [ ] Tap "Add New Face"
  - [ ] Enter a name
  - [ ] Capture all 5 poses
  - [ ] See success message
  - [ ] Check database count updated

- [ ] **Test 2: View Database**
  - [ ] Tap "View Database"
  - [ ] See your enrolled face
  - [ ] Check pose count shows 5

- [ ] **Test 3: Verify Face**
  - [ ] Tap "Verify Face"
  - [ ] Position face in camera
  - [ ] Tap "Verify"
  - [ ] Should recognize you with >60% confidence

- [ ] **Test 4: Delete Face**
  - [ ] Tap "Delete Face"
  - [ ] Select a face
  - [ ] Confirm deletion
  - [ ] Verify face removed from database

---

## 🎯 Success Criteria

✅ **Your app is working correctly if:**

1. Model conversion completed without errors
2. App builds and installs successfully
3. Camera preview works
4. Face detection works (green boxes on faces)
5. Can enroll a face with all 5 poses
6. Enrolled face appears in database
7. Verification recognizes enrolled faces
8. Can delete faces from database
9. No crashes during normal use

---

## 🐛 Common Issues & Quick Fixes

### Issue: "Model not found" error
**Fix:** Run `python convert_model_to_mobile.py` again

### Issue: Face not detected during enrollment
**Fix:** 
- Improve lighting
- Face camera directly
- Remove glasses/masks
- Ensure face is centered

### Issue: Low verification accuracy
**Fix:**
- Lower threshold in `FaceRecognitionModel.kt` (line 19): `0.5f` instead of `0.6f`
- Re-enroll with better lighting
- Capture more varied poses

### Issue: App crashes on launch
**Fix:**
1. Check Logcat for specific error
2. Verify `model.pt` exists in assets folder
3. Clean and rebuild project

### Issue: Camera preview is black
**Fix:**
- Grant camera permission in device settings
- Restart app
- Check device camera works in other apps

### Issue: Gradle sync fails
**Fix:**
1. Check internet connection
2. File → Invalidate Caches → Invalidate and Restart
3. Update Android Studio to latest version

---

## 📊 Performance Benchmarks

After setup, test performance:

- [ ] Enrollment time: < 30 seconds for 5 poses
- [ ] Face detection: < 100ms per frame
- [ ] Verification time: < 2 seconds total
- [ ] App size: < 70 MB
- [ ] No lag in camera preview

---

## 🎓 Learning Resources

**New to Android Development?**
- Official docs: https://developer.android.com/docs
- Kotlin basics: https://kotlinlang.org/docs/getting-started.html
- CameraX tutorial: https://developer.android.com/training/camerax

**Debugging Help:**
- Logcat guide: https://developer.android.com/studio/debug/am-logcat
- Build troubleshooting: https://developer.android.com/studio/build/dependencies

---

## 🎉 Completion

Once all items are checked:

✅ **Congratulations!** Your Android Face ID app is fully functional!

**Next steps:**
- Share APK with others
- Customize UI colors/themes
- Adjust similarity threshold
- Add more features (e.g., multiple faces per person)

---

## 📞 Need Help?

**Check these in order:**
1. ✅ This checklist - did you complete all steps?
2. 📖 QUICKSTART.md - quick reference guide
3. 📚 README.md - detailed documentation
4. 📋 Logcat in Android Studio - error messages
5. 🔍 Google the specific error message

**Most issues are:**
- Missing model file (forgot to run conversion script)
- Camera permissions not granted
- Gradle sync needs internet connection
- Model path mismatch

---

## 💾 Save Your Progress

- [ ] Commit Android_App to git
- [ ] Backup converted model.pt
- [ ] Save any custom changes you made
- [ ] Document threshold adjustments

---

**Happy developing! 🚀**
