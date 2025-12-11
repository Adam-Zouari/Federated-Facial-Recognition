# Face ID Android App

Android equivalent of the Python Face ID application for face enrollment and verification.

## Features

- 📷 **Face Enrollment**: Capture 5 different poses for robust recognition
- 🔍 **Face Verification**: Real-time face verification against stored database
- 💾 **Local Database**: Store face embeddings securely on device
- 🤖 **ML Kit Integration**: Google ML Kit for accurate face detection
- 🚀 **PyTorch Mobile**: Run your trained model on Android

## Architecture

```
FaceIDApp/
├── app/
│   ├── src/main/
│   │   ├── java/com/faceid/
│   │   │   ├── MainActivity.kt              # Main UI with buttons
│   │   │   ├── EnrollmentActivity.kt        # Face enrollment with poses
│   │   │   ├── VerificationActivity.kt      # Face verification
│   │   │   ├── DatabaseActivity.kt          # View/manage database
│   │   │   ├── ml/
│   │   │   │   ├── FaceRecognitionModel.kt  # PyTorch model inference
│   │   │   │   └── FaceDetector.kt          # ML Kit face detection
│   │   │   ├── data/
│   │   │   │   ├── FaceDatabase.kt          # Database management
│   │   │   │   └── FaceEmbedding.kt         # Data models
│   │   │   └── utils/
│   │   │       ├── ImageUtils.kt            # Image preprocessing
│   │   │       └── PermissionUtils.kt       # Camera permissions
│   │   ├── res/
│   │   │   ├── layout/                      # UI layouts
│   │   │   ├── values/                      # Strings, colors, themes
│   │   │   └── drawable/                    # Icons and resources
│   │   └── assets/
│   │       └── model.pt                     # TorchScript model (converted)
│   └── build.gradle
├── build.gradle
└── settings.gradle
```

## Setup Instructions

### 1. Convert Your PyTorch Model to TorchScript

Run the provided conversion script:

```bash
python convert_model_to_mobile.py --model checkpoints/local/vggface2_weak/best_model.pth --output Android_App/app/src/main/assets/model.pt
```

### 2. Open in Android Studio

1. Install [Android Studio](https://developer.android.com/studio)
2. Open the `Android_App` folder as a project
3. Wait for Gradle sync to complete

### 3. Build and Run

1. Connect an Android device or start an emulator (API 24+)
2. Click "Run" or press Shift+F10
3. Grant camera permissions when prompted

## Dependencies

The app uses:
- **PyTorch Mobile**: For model inference
- **Google ML Kit**: For face detection
- **CameraX**: For camera functionality
- **Gson**: For JSON database serialization
- **Material Design**: For modern UI

All dependencies are automatically downloaded by Gradle.

## Usage

### Enrolling a New Face

1. Tap "Add New Face"
2. Enter your name
3. Follow on-screen instructions to capture 5 poses:
   - Front view
   - Slight left turn
   - Slight right turn
   - Slight upward tilt
   - Slight downward tilt
4. Face is saved to local database

### Verifying a Face

1. Tap "Verify Face"
2. Look at the camera
3. The app will identify you if you're in the database
4. Shows confidence score

### Managing Database

- **View Database**: See all registered faces
- **Delete Face**: Remove specific faces from database

## Model Requirements

Your PyTorch model must:
- Accept input of shape `[1, 3, 224, 224]` (or your configured IMG_SIZE)
- Return embeddings when called with `return_embedding=True`
- Use normalized embeddings (L2 normalization)

## Similarity Threshold

Default threshold is **0.6** (60% similarity). You can adjust this in `FaceRecognitionModel.kt`:

```kotlin
companion object {
    private const val SIMILARITY_THRESHOLD = 0.6f
}
```

## Permissions

The app requires:
- `CAMERA`: For face capture
- `INTERNET`: For ML Kit models (downloaded on first run)

## Performance Tips

1. **Model Size**: Use MobileNetV2 for optimal mobile performance
2. **Image Resolution**: 224x224 is a good balance between speed and accuracy
3. **Face Detection**: ML Kit runs on-device and is very fast
4. **Storage**: Embeddings are small (~128 floats each)

## Troubleshooting

### Model Not Loading
- Ensure `model.pt` is in `app/src/main/assets/`
- Check conversion script ran successfully
- Verify model is TorchScript format, not regular .pth

### Face Not Detected
- Ensure good lighting
- Face should be clearly visible
- Check camera permissions are granted

### Low Accuracy
- Increase similarity threshold for stricter verification
- Capture more poses during enrollment
- Improve lighting conditions

## Technical Details

### Face Detection
Uses Google ML Kit Face Detection API:
- Detects face landmarks
- Returns bounding boxes
- Works in real-time

### Embedding Extraction
1. Detect face with ML Kit
2. Crop and add 20% padding
3. Resize to 224x224
4. Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
5. Run through PyTorch model
6. Extract and normalize embedding

### Face Matching
- Uses cosine similarity between embeddings
- Compares with all stored poses
- Uses maximum similarity for matching
- Threshold-based verification

### Database Format
JSON file stored in app's private storage:
```json
{
  "John Doe": {
    "embeddings": [[0.123, -0.456, ...], ...],
    "dateAdded": "2025-12-11 10:30:00",
    "numPoses": 5
  }
}
```

## Building APK

```bash
cd Android_App
./gradlew assembleRelease
```

APK will be in `app/build/outputs/apk/release/`

## Minimum Requirements

- Android 7.0 (API 24) or higher
- Device with camera
- ~50MB storage for app and models

## License

Same as parent project

## Support

For issues specific to Android implementation, check:
1. Android Studio build logs
2. Logcat for runtime errors
3. Ensure model conversion was successful
