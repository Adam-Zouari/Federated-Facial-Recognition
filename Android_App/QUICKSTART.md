# Face ID Android App - Quick Start Guide

## 🚀 Quick Setup (5 Minutes)

### Step 1: Convert Your Model

In your project root, run:

```bash
python convert_model_to_mobile.py
```

This will:
- Convert your trained model to TorchScript format
- Save it to `Android_App/app/src/main/assets/model.pt`
- Show you the model size and details

**Note:** The script uses your best model at `checkpoints/local/vggface2_weak/best_model.pth` by default.

To use a different model:
```bash
python convert_model_to_mobile.py --model path/to/your/model.pth
```

### Step 2: Open in Android Studio

1. **Download Android Studio** (if not installed): https://developer.android.com/studio
2. **Open the project**: 
   - Launch Android Studio
   - Click "Open"
   - Navigate to `Android_App` folder
   - Click "OK"
3. **Wait for Gradle sync** (first time takes a few minutes)

### Step 3: Run the App

1. **Connect device or start emulator**:
   - Physical device: Enable USB debugging in Developer Options
   - Emulator: Click "Device Manager" → "Create Device" (API 24+)

2. **Click the green "Run" button** (▶) or press `Shift+F10`

3. **Grant camera permission** when prompted

That's it! 🎉

---

## 📱 Using the App

### Enrolling Your Face

1. Tap **"Add New Face"**
2. Enter your name
3. Capture 5 poses:
   - Front view
   - Turn left
   - Turn right
   - Look up
   - Look down
4. Tap the screen to capture each pose
5. Done! Your face is saved

### Verifying Your Identity

1. Tap **"Verify Face"**
2. Position your face in the camera
3. Tap **"Verify"**
4. Get instant results with confidence score

### Managing Database

- **View Database**: See all registered faces
- **Delete Face**: Remove people from the database

---

## 🔧 Troubleshooting

### "Model not found" error

Make sure you ran the conversion script:
```bash
python convert_model_to_mobile.py
```

Check that `model.pt` exists in:
```
Android_App/app/src/main/assets/model.pt
```

### Gradle sync failed

1. Check internet connection (downloads dependencies)
2. File → Invalidate Caches → Invalidate and Restart
3. Try again

### Face not detected

- Ensure good lighting
- Face camera directly
- Remove glasses if detection fails
- Check camera permissions are granted

### Low accuracy

Adjust threshold in `FaceRecognitionModel.kt`:
```kotlin
companion object {
    private const val SIMILARITY_THRESHOLD = 0.6f  // Lower = more lenient
}
```

Recommended values:
- `0.5` - Very lenient (more false positives)
- `0.6` - Balanced (default)
- `0.7` - Strict (fewer false positives)

---

## 📊 Technical Comparison

| Feature | Python App | Android App |
|---------|-----------|-------------|
| Face Detection | MTCNN / Haar Cascade | Google ML Kit |
| Model Runtime | PyTorch CPU/GPU | PyTorch Mobile |
| Camera | OpenCV | CameraX |
| GUI | Tkinter | Material Design |
| Database | JSON file | JSON file |
| Platforms | Windows/Mac/Linux | Android 7.0+ |

---

## 🎯 Model Requirements

Your model MUST:
- Accept input: `[1, 3, 224, 224]`
- Return embeddings via `return_embedding=True`
- Use L2-normalized embeddings

The conversion script handles this automatically!

---

## 📦 What's Included

```
Android_App/
├── app/
│   ├── src/main/java/com/faceid/
│   │   ├── MainActivity.kt              ✓ Main UI
│   │   ├── EnrollmentActivity.kt        ✓ Face enrollment
│   │   ├── VerificationActivity.kt      ✓ Face verification
│   │   ├── DatabaseActivity.kt          ✓ Database management
│   │   ├── ml/
│   │   │   ├── FaceRecognitionModel.kt  ✓ Model inference
│   │   │   └── FaceDetector.kt          ✓ Face detection
│   │   └── data/
│   │       └── FaceDatabase.kt          ✓ Database
│   └── src/main/assets/
│       └── model.pt                     ← Your converted model
└── README.md                            ✓ Full documentation
```

---

## 💡 Pro Tips

1. **Better Lighting = Better Results**
   - Face the light source
   - Avoid backlighting
   - Use natural light when possible

2. **Consistent Angles**
   - Capture enrollment poses consistently
   - Verification works best with similar angles

3. **Multiple Poses Help**
   - The 5-pose enrollment captures face variations
   - Helps with different lighting and angles

4. **Model Performance**
   - MobileNetV2 is optimized for mobile
   - Runs real-time on most devices
   - ~50MB model size

5. **Database Backup**
   - Database stored in app private storage
   - Automatically saved after each change
   - Uninstalling app deletes database

---

## 🔐 Privacy & Security

- ✅ All processing happens **on-device**
- ✅ No internet required (except ML Kit initial download)
- ✅ Face embeddings stored locally
- ✅ No data sent to servers
- ✅ Embeddings are encrypted by Android

---

## 🛠️ Building Release APK

```bash
cd Android_App
./gradlew assembleRelease
```

APK location:
```
app/build/outputs/apk/release/app-release-unsigned.apk
```

For signed APK, configure signing in `app/build.gradle.kts`

---

## 📞 Support

**Model conversion issues?**
- Check Python environment has PyTorch installed
- Verify model path is correct
- Check checkpoint format

**Android Studio issues?**
- Update to latest version
- Sync Gradle files
- Check SDK is installed (API 24+)

**Runtime issues?**
- Check Logcat for errors
- Verify camera permissions
- Ensure model.pt exists in assets

---

## ✨ Enjoy Your Face ID App!

You now have a fully functional Android face recognition app equivalent to your Python desktop version, with modern UI and optimized performance!
