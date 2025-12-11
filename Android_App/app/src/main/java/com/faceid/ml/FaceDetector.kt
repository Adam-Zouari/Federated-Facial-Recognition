package com.faceid.ml

import android.graphics.Bitmap
import android.util.Log
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.coroutines.suspendCoroutine

class FaceDetector {
    
    companion object {
        private const val TAG = "FaceDetector"
        private const val PADDING_PERCENT = 0.2f  // 20% padding around detected face
        private const val MAX_IMAGE_DIMENSION = 1024  // Max width/height for face detection
        
        /**
         * Resize bitmap if it's too large for efficient face detection
         */
        private fun resizeBitmap(bitmap: Bitmap): Bitmap {
            val maxDimension = maxOf(bitmap.width, bitmap.height)
            if (maxDimension <= MAX_IMAGE_DIMENSION) {
                return bitmap
            }
            
            val scale = MAX_IMAGE_DIMENSION.toFloat() / maxDimension
            val newWidth = (bitmap.width * scale).toInt()
            val newHeight = (bitmap.height * scale).toInt()
            
            Log.d(TAG, "Resizing image from ${bitmap.width}x${bitmap.height} to ${newWidth}x${newHeight}")
            return Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true)
        }
    }

    private val detector = FaceDetection.getClient(
        FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
            .setContourMode(FaceDetectorOptions.CONTOUR_MODE_NONE)
            .setMinFaceSize(0.05f)  // Very small minimum (5% of image)
            .build()
    )

    /**
     * Detect face in bitmap and return cropped face with padding
     * 
     * @param bitmap Input image
     * @return Cropped face bitmap or null if no face detected
     */
    suspend fun detectAndCropFace(bitmap: Bitmap): Bitmap? = suspendCoroutine { continuation ->
        // Resize bitmap if too large
        val resizedBitmap = resizeBitmap(bitmap)
        
        Log.d(TAG, "Processing image: ${resizedBitmap.width}x${resizedBitmap.height}, hasAlpha=${resizedBitmap.hasAlpha()}, config=${resizedBitmap.config}")
        
        // Ensure bitmap is in ARGB_8888 format (required by ML Kit)
        val argbBitmap = if (resizedBitmap.config != Bitmap.Config.ARGB_8888) {
            Log.d(TAG, "Converting bitmap to ARGB_8888")
            resizedBitmap.copy(Bitmap.Config.ARGB_8888, false)
        } else {
            resizedBitmap
        }
        
        val image = InputImage.fromBitmap(argbBitmap, 0)
        
        detector.process(image)
            .addOnSuccessListener { faces ->
                if (faces.isEmpty()) {
                    Log.d(TAG, "No faces detected")
                    continuation.resume(null)
                    return@addOnSuccessListener
                }

                // Use the largest face
                val face = faces.maxByOrNull { it.boundingBox.width() * it.boundingBox.height() }
                    ?: run {
                        continuation.resume(null)
                        return@addOnSuccessListener
                    }

                try {
                    // Calculate scale factor to map face coordinates back to original bitmap
                    val scaleX = bitmap.width.toFloat() / resizedBitmap.width
                    val scaleY = bitmap.height.toFloat() / resizedBitmap.height
                    
                    val croppedFace = cropFaceWithPadding(bitmap, face, scaleX, scaleY)
                    Log.d(TAG, "Face detected and cropped successfully")
                    continuation.resume(croppedFace)
                } catch (e: Exception) {
                    Log.e(TAG, "Error cropping face: ${e.message}", e)
                    continuation.resume(null)
                }
            }
            .addOnFailureListener { e ->
                Log.e(TAG, "Face detection failed: ${e.message}", e)
                continuation.resumeWithException(e)
            }
    }

    /**
     * Detect all faces in bitmap
     * 
     * @param bitmap Input image
     * @return List of detected faces
     */
    suspend fun detectFaces(bitmap: Bitmap): List<Face> = suspendCoroutine { continuation ->
        val image = InputImage.fromBitmap(bitmap, 0)
        
        detector.process(image)
            .addOnSuccessListener { faces ->
                Log.d(TAG, "Detected ${faces.size} face(s)")
                continuation.resume(faces)
            }
            .addOnFailureListener { e ->
                Log.e(TAG, "Face detection failed: ${e.message}", e)
                continuation.resumeWithException(e)
            }
    }

    /**
     * Crop face from bitmap with padding, scaling coordinates if needed
     */
    private fun cropFaceWithPadding(bitmap: Bitmap, face: Face, scaleX: Float = 1f, scaleY: Float = 1f): Bitmap {
        val boundingBox = face.boundingBox
        
        // Scale coordinates to original bitmap size
        val scaledLeft = (boundingBox.left * scaleX).toInt()
        val scaledTop = (boundingBox.top * scaleY).toInt()
        val scaledRight = (boundingBox.right * scaleX).toInt()
        val scaledBottom = (boundingBox.bottom * scaleY).toInt()
        
        // Calculate padding
        val width = scaledRight - scaledLeft
        val height = scaledBottom - scaledTop
        val padding = (PADDING_PERCENT * maxOf(width, height)).toInt()

        // Apply padding while keeping within bitmap bounds
        val left = maxOf(0, scaledLeft - padding)
        val top = maxOf(0, scaledTop - padding)
        val right = minOf(bitmap.width, scaledRight + padding)
        val bottom = minOf(bitmap.height, scaledBottom + padding)

        val croppedWidth = right - left
        val croppedHeight = bottom - top

        return Bitmap.createBitmap(bitmap, left, top, croppedWidth, croppedHeight)
    }

    /**
     * Check if face is detected in bitmap
     */
    suspend fun hasFace(bitmap: Bitmap): Boolean {
        val faces = detectFaces(bitmap)
        return faces.isNotEmpty()
    }

    /**
     * Clean up resources
     */
    fun close() {
        detector.close()
    }
}
