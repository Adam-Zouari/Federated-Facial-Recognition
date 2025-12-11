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
import com.faceid.databinding.ActivityVerificationBinding
import com.faceid.data.FaceDatabase
import com.faceid.ml.FaceDetector
import com.faceid.ml.FaceRecognitionModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class VerificationActivity : AppCompatActivity() {
    
    private lateinit var binding: ActivityVerificationBinding
    private lateinit var faceDetector: FaceDetector
    private lateinit var faceModel: FaceRecognitionModel
    private lateinit var faceDatabase: FaceDatabase
    private lateinit var cameraExecutor: ExecutorService
    
    private var imageCapture: ImageCapture? = null
    private val isDetecting = AtomicBoolean(false)
    
    companion object {
        private const val TAG = "VerificationActivity"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityVerificationBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Initialize
        faceDetector = FaceDetector()
        faceModel = FaceRecognitionModel(this)
        faceDatabase = FaceDatabase.getInstance(this)
        cameraExecutor = Executors.newSingleThreadExecutor()

        // Show instructions
        AlertDialog.Builder(this)
            .setTitle("Instructions")
            .setMessage("Look at the camera and tap 'Verify' to check your identity.")
            .setPositiveButton("OK") { _, _ -> setupCamera() }
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

        // Verify button
        binding.btnVerify.setOnClickListener {
            verifyFace()
        }
    }

    private fun verifyFace() {
        val imageCapture = imageCapture ?: return

        binding.btnVerify.isEnabled = false
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
                            this@VerificationActivity,
                            "Capture failed. Try again.",
                            Toast.LENGTH_SHORT
                        ).show()
                        binding.btnVerify.isEnabled = true
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
                imageProxy.close()
                
                // Rotate bitmap if needed
                val rotatedBitmap = if (rotationDegrees != 0) {
                    rotateBitmap(bitmap, rotationDegrees.toFloat())
                } else {
                    bitmap
                }

                binding.tvStatus.text = "Detecting face..."

                // Detect and crop face
                val faceBitmap = withContext(Dispatchers.Default) {
                    faceDetector.detectAndCropFace(rotatedBitmap)
                }

                if (faceBitmap == null) {
                    withContext(Dispatchers.Main) {
                        Toast.makeText(
                            this@VerificationActivity,
                            "No face detected. Try again.",
                            Toast.LENGTH_SHORT
                        ).show()
                        binding.btnVerify.isEnabled = true
                        binding.tvStatus.text = ""
                    }
                    return@launch
                }

                withContext(Dispatchers.Main) {
                    binding.tvStatus.text = "Extracting features..."
                }

                // Extract embedding
                val testEmbedding = withContext(Dispatchers.Default) {
                    faceModel.extractEmbedding(faceBitmap)
                }

                if (testEmbedding == null) {
                    withContext(Dispatchers.Main) {
                        Toast.makeText(
                            this@VerificationActivity,
                            "Failed to extract face features. Try again.",
                            Toast.LENGTH_SHORT
                        ).show()
                        binding.btnVerify.isEnabled = true
                        binding.tvStatus.text = ""
                    }
                    return@launch
                }

                withContext(Dispatchers.Main) {
                    binding.tvStatus.text = "Comparing with database..."
                }

                // Compare with database
                val (bestMatch, bestSimilarity) = withContext(Dispatchers.Default) {
                    var bestName: String? = null
                    var bestScore = -1f

                    faceDatabase.getAllFaces().forEach { face ->
                        val similarity = faceModel.verifyFace(testEmbedding, face.embeddings)
                        if (similarity > bestScore) {
                            bestScore = similarity
                            bestName = face.name
                        }
                    }

                    Pair(bestName, bestScore)
                }

                // Show result
                withContext(Dispatchers.Main) {
                    binding.tvStatus.text = ""
                    binding.btnVerify.isEnabled = true

                    if (bestSimilarity >= FaceRecognitionModel.SIMILARITY_THRESHOLD) {
                        // Access granted
                        AlertDialog.Builder(this@VerificationActivity)
                            .setTitle("✅ Access Granted")
                            .setMessage(
                                "Welcome, $bestMatch!\n\n" +
                                "Confidence: ${(bestSimilarity * 100).toInt()}%"
                            )
                            .setPositiveButton("OK") { _, _ -> finish() }
                            .setCancelable(false)
                            .show()
                    } else {
                        // Unknown face
                        AlertDialog.Builder(this@VerificationActivity)
                            .setTitle("❌ Unknown Face")
                            .setMessage(
                                "Face not recognized.\n\n" +
                                "Best match: $bestMatch (${(bestSimilarity * 100).toInt()}%)\n" +
                                "Threshold: ${(FaceRecognitionModel.SIMILARITY_THRESHOLD * 100).toInt()}%"
                            )
                            .setPositiveButton("Try Again") { _, _ -> }
                            .setNegativeButton("Cancel") { _, _ -> finish() }
                            .show()
                    }
                }

            } catch (e: Exception) {
                Log.e(TAG, "Error processing image: ${e.message}", e)
                withContext(Dispatchers.Main) {
                    Toast.makeText(
                        this@VerificationActivity,
                        "Error processing image",
                        Toast.LENGTH_SHORT
                    ).show()
                    binding.btnVerify.isEnabled = true
                    binding.tvStatus.text = ""
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
