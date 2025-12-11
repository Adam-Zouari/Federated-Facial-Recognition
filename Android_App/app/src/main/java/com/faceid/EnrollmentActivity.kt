package com.faceid

import android.graphics.Bitmap
import android.graphics.RectF
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.faceid.databinding.ActivityEnrollmentBinding
import com.faceid.data.FaceDatabase
import com.faceid.ml.FaceDetector
import com.faceid.ml.FaceRecognitionModel
import com.google.android.material.textfield.TextInputEditText
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class EnrollmentActivity : AppCompatActivity() {
    
    private lateinit var binding: ActivityEnrollmentBinding
    private lateinit var faceDetector: FaceDetector
    private lateinit var faceModel: FaceRecognitionModel
    private lateinit var faceDatabase: FaceDatabase
    private lateinit var cameraExecutor: ExecutorService
    
    private var imageCapture: ImageCapture? = null
    private var currentPoseIndex = 0
    private val capturedEmbeddings = mutableListOf<FloatArray>()
    private var enrollmentName = ""
    private val isDetecting = AtomicBoolean(false)
    
    private val poses = listOf(
        "Look straight at the camera",
        "Turn your head slightly LEFT",
        "Turn your head slightly RIGHT",
        "Tilt your head slightly UP",
        "Tilt your head slightly DOWN"
    )
    
    companion object {
        private const val TAG = "EnrollmentActivity"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        try {
            binding = ActivityEnrollmentBinding.inflate(layoutInflater)
            setContentView(binding.root)

            // Initialize
            faceDetector = FaceDetector()
            faceModel = FaceRecognitionModel(this)
            faceDatabase = FaceDatabase.getInstance(this)
            cameraExecutor = Executors.newSingleThreadExecutor()

            // Ask for name first
            askForName()
        } catch (e: Exception) {
            Toast.makeText(
                this,
                "Error initializing: ${e.message}",
                Toast.LENGTH_LONG
            ).show()
            e.printStackTrace()
            finish()
        }
    }

    private fun askForName() {
        val input = TextInputEditText(this)
        input.hint = "Enter your name"

        AlertDialog.Builder(this)
            .setTitle("Add New Face")
            .setMessage("Please enter your name:")
            .setView(input)
            .setPositiveButton("Start") { _, _ ->
                val name = input.text.toString().trim()
                if (name.isNotEmpty()) {
                    enrollmentName = name
                    
                    // Check if name exists
                    if (faceDatabase.contains(name)) {
                        AlertDialog.Builder(this)
                            .setTitle("Name Exists")
                            .setMessage("$name already exists. Do you want to update their face data?")
                            .setPositiveButton("Yes") { _, _ -> startEnrollment() }
                            .setNegativeButton("No") { _, _ -> finish() }
                            .setCancelable(false)
                            .show()
                    } else {
                        startEnrollment()
                    }
                } else {
                    Toast.makeText(this, "Name cannot be empty", Toast.LENGTH_SHORT).show()
                    finish()
                }
            }
            .setNegativeButton("Cancel") { _, _ -> finish() }
            .setCancelable(false)
            .show()
    }

    private fun startEnrollment() {
        // Show instructions
        AlertDialog.Builder(this)
            .setTitle("Instructions")
            .setMessage(
                "We will capture 5 different poses of your face.\n\n" +
                "Follow the on-screen instructions for each pose.\n\n" +
                "Tap the screen to capture each pose."
            )
            .setPositiveButton("OK") { _, _ ->
                setupCamera()
                updatePoseInstruction()
            }
            .setCancelable(false)
            .show()
    }

    private fun setupCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            // Preview
            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(binding.previewView.surfaceProvider)
                }

            // Image capture
            imageCapture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build()

            // Image analysis for real-time face detection
            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        detectFaceInPreview(imageProxy)
                    }
                }

            // Select front camera
            val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this,
                    cameraSelector,
                    preview,
                    imageCapture,
                    imageAnalyzer
                )
            } catch (e: Exception) {
                Log.e(TAG, "Camera binding failed", e)
                Toast.makeText(this, "Camera initialization failed", Toast.LENGTH_SHORT).show()
                finish()
            }
        }, ContextCompat.getMainExecutor(this))

        // Capture button
        binding.btnCapture.setOnClickListener {
            capturePhoto()
        }
    }

    private fun updatePoseInstruction() {
        if (currentPoseIndex < poses.size) {
            binding.tvPoseInstruction.text = poses[currentPoseIndex]
            binding.tvPoseCounter.text = "Pose ${currentPoseIndex + 1}/${poses.size}"
        }
    }

    private fun capturePhoto() {
        val imageCapture = imageCapture ?: return

        binding.btnCapture.isEnabled = false
        binding.tvStatus.text = "Processing..."

        imageCapture.takePicture(
            cameraExecutor,
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(image: ImageProxy) {
                    processCapturedImage(image)
                }

                override fun onError(exception: ImageCaptureException) {
                    Log.e(TAG, "Photo capture failed: ${exception.message}", exception)
                    runOnUiThread {
                        Toast.makeText(
                            this@EnrollmentActivity,
                            "Capture failed. Try again.",
                            Toast.LENGTH_SHORT
                        ).show()
                        binding.btnCapture.isEnabled = true
                        binding.tvStatus.text = ""
                    }
                }
            }
        )
    }

    private fun processCapturedImage(imageProxy: ImageProxy) {
        lifecycleScope.launch {
            try {
                val bitmap = imageProxy.toBitmap()
                val rotationDegrees = imageProxy.imageInfo.rotationDegrees
                Log.d(TAG, "Captured image: ${bitmap.width}x${bitmap.height}, rotation: $rotationDegrees")
                imageProxy.close()
                
                // Rotate bitmap if needed
                val rotatedBitmap = if (rotationDegrees != 0) {
                    rotateBitmap(bitmap, rotationDegrees.toFloat())
                } else {
                    bitmap
                }
                Log.d(TAG, "Final image: ${rotatedBitmap.width}x${rotatedBitmap.height}")

                // Detect and crop face
                val faceBitmap = withContext(Dispatchers.Default) {
                    Log.d(TAG, "Starting face detection...")
                    val result = faceDetector.detectAndCropFace(rotatedBitmap)
                    Log.d(TAG, "Face detection result: ${if (result != null) "Success" else "No face found"}")
                    result
                }

                if (faceBitmap == null) {
                    withContext(Dispatchers.Main) {
                        Toast.makeText(
                            this@EnrollmentActivity,
                            "No face detected. Try again.",
                            Toast.LENGTH_SHORT
                        ).show()
                        binding.btnCapture.isEnabled = true
                        binding.tvStatus.text = ""
                    }
                    return@launch
                }

                // Extract embedding
                val embedding = withContext(Dispatchers.Default) {
                    faceModel.extractEmbedding(faceBitmap)
                }

                if (embedding == null) {
                    withContext(Dispatchers.Main) {
                        Toast.makeText(
                            this@EnrollmentActivity,
                            "Failed to extract face features. Try again.",
                            Toast.LENGTH_SHORT
                        ).show()
                        binding.btnCapture.isEnabled = true
                        binding.tvStatus.text = ""
                    }
                    return@launch
                }

                // Save embedding
                capturedEmbeddings.add(embedding)
                Log.d(TAG, "Captured pose ${currentPoseIndex + 1}/${poses.size}")

                withContext(Dispatchers.Main) {
                    currentPoseIndex++
                    
                    if (currentPoseIndex < poses.size) {
                        // Next pose
                        updatePoseInstruction()
                        binding.btnCapture.isEnabled = true
                        binding.tvStatus.text = "✓ Pose captured!"
                        
                        Toast.makeText(
                            this@EnrollmentActivity,
                            "Pose ${currentPoseIndex}/${poses.size} captured!",
                            Toast.LENGTH_SHORT
                        ).show()
                    } else {
                        // All poses captured, save to database
                        saveToDatabase()
                    }
                }

            } catch (e: Exception) {
                Log.e(TAG, "Error processing image: ${e.message}", e)
                withContext(Dispatchers.Main) {
                    Toast.makeText(
                        this@EnrollmentActivity,
                        "Error processing image",
                        Toast.LENGTH_SHORT
                    ).show()
                    binding.btnCapture.isEnabled = true
                    binding.tvStatus.text = ""
                }
            }
        }
    }

    private fun saveToDatabase() {
        binding.tvStatus.text = "Saving to database..."
        
        lifecycleScope.launch {
            val success = withContext(Dispatchers.IO) {
                faceDatabase.addFace(enrollmentName, capturedEmbeddings)
            }

            withContext(Dispatchers.Main) {
                if (success) {
                    AlertDialog.Builder(this@EnrollmentActivity)
                        .setTitle("Success")
                        .setMessage(
                            "✓ $enrollmentName has been successfully enrolled!\n\n" +
                            "Captured ${capturedEmbeddings.size} poses."
                        )
                        .setPositiveButton("OK") { _, _ -> finish() }
                        .setCancelable(false)
                        .show()
                } else {
                    Toast.makeText(
                        this@EnrollmentActivity,
                        "Failed to save to database",
                        Toast.LENGTH_LONG
                    ).show()
                    finish()
                }
            }
        }
    }
    
    private fun rotateBitmap(bitmap: Bitmap, degrees: Float): Bitmap {
        val matrix = android.graphics.Matrix()
        matrix.postRotate(degrees)
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }
    
    private fun detectFaceInPreview(imageProxy: ImageProxy) {
        // Skip if already detecting
        if (!isDetecting.compareAndSet(false, true)) {
            imageProxy.close()
            return
        }
        
        lifecycleScope.launch(Dispatchers.Default) {
            try {
                val bitmap = imageProxy.toBitmap()
                val rotationDegrees = imageProxy.imageInfo.rotationDegrees
                imageProxy.close()
                
                // Rotate bitmap if needed
                val rotatedBitmap = if (rotationDegrees != 0) {
                    rotateBitmap(bitmap, rotationDegrees.toFloat())
                } else {
                    bitmap
                }
                
                // Detect faces
                val faces = faceDetector.detectFaces(rotatedBitmap)
                
                withContext(Dispatchers.Main) {
                    if (faces.isNotEmpty()) {
                        // Get largest face
                        val face = faces.maxByOrNull { it.boundingBox.width() * it.boundingBox.height() }
                        face?.let {
                            val rect = RectF(
                                it.boundingBox.left.toFloat(),
                                it.boundingBox.top.toFloat(),
                                it.boundingBox.right.toFloat(),
                                it.boundingBox.bottom.toFloat()
                            )
                            // Scale rect to preview size
                            val scaleX = binding.previewView.width.toFloat() / rotatedBitmap.width
                            val scaleY = binding.previewView.height.toFloat() / rotatedBitmap.height
                            rect.left *= scaleX
                            rect.right *= scaleX
                            rect.top *= scaleY
                            rect.bottom *= scaleY
                            
                            binding.faceOverlay.setFaceRect(rect)
                        }
                    } else {
                        binding.faceOverlay.setFaceRect(null)
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error in preview detection: ${e.message}")
            } finally {
                isDetecting.set(false)
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        faceDetector.close()
        faceModel.close()
    }
}

// Extension function to convert ImageProxy to Bitmap
fun ImageProxy.toBitmap(): Bitmap {
    val buffer = planes[0].buffer
    val bytes = ByteArray(buffer.remaining())
    buffer.get(bytes)
    return android.graphics.BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
}
