package com.faceid.ml

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream
import kotlin.math.sqrt

class FaceRecognitionModel(context: Context) {
    private val model: Module
    
    companion object {
        private const val TAG = "FaceRecognitionModel"
        private const val MODEL_FILE = "model.pt"
        
        // Image preprocessing constants (must match training config)
        private val NORMALIZE_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val NORMALIZE_STD = floatArrayOf(0.229f, 0.224f, 0.225f)
        private const val IMG_SIZE = 128  // Model expects 128x128 input
        
        // Similarity threshold for verification (70%)
        const val SIMILARITY_THRESHOLD = 0.7f
    }

    init {
        // Load model from assets
        model = try {
            val assetPath = assetFilePath(context, MODEL_FILE)
            Module.load(assetPath)
        } catch (e: Exception) {
            Log.e(TAG, "Error loading model: ${e.message}", e)
            throw RuntimeException("Failed to load PyTorch model. Ensure model.pt is in assets/", e)
        }
        Log.d(TAG, "Model loaded successfully")
    }

    /**
     * Extract face embedding from a bitmap image
     * 
     * @param faceBitmap Cropped face bitmap (will be resized to IMG_SIZE x IMG_SIZE)
     * @return FloatArray embedding (128-dim) or null if extraction fails
     */
    fun extractEmbedding(faceBitmap: Bitmap): FloatArray? {
        return try {
            // Resize to model input size
            val resizedBitmap = Bitmap.createScaledBitmap(
                faceBitmap,
                IMG_SIZE,
                IMG_SIZE,
                true
            )

            // Convert to tensor with normalization
            val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                resizedBitmap,
                NORMALIZE_MEAN,
                NORMALIZE_STD
            )

            // Run inference
            val outputTensor = model.forward(IValue.from(inputTensor)).toTensor()
            
            // Extract embedding (model returns embeddings directly for mobile version)
            val embedding = outputTensor.dataAsFloatArray
            
            // Normalize embedding (L2 normalization)
            normalizeEmbedding(embedding)
            
            Log.d(TAG, "Embedding extracted successfully (dim: ${embedding.size})")
            embedding
        } catch (e: Exception) {
            Log.e(TAG, "Error extracting embedding: ${e.message}", e)
            null
        }
    }

    /**
     * Calculate cosine similarity between two embeddings
     * 
     * @param embedding1 First embedding
     * @param embedding2 Second embedding
     * @return Cosine similarity score (0 to 1)
     */
    fun cosineSimilarity(embedding1: FloatArray, embedding2: FloatArray): Float {
        if (embedding1.size != embedding2.size) {
            Log.e(TAG, "Embedding size mismatch: ${embedding1.size} vs ${embedding2.size}")
            return 0f
        }

        var dotProduct = 0f
        var norm1 = 0f
        var norm2 = 0f

        for (i in embedding1.indices) {
            dotProduct += embedding1[i] * embedding2[i]
            norm1 += embedding1[i] * embedding1[i]
            norm2 += embedding2[i] * embedding2[i]
        }

        val denominator = sqrt(norm1) * sqrt(norm2)
        return if (denominator > 0) dotProduct / denominator else 0f
    }

    /**
     * Verify if test embedding matches any stored embeddings
     * 
     * @param testEmbedding Embedding to test
     * @param storedEmbeddings List of stored embeddings to compare against
     * @return Maximum similarity score
     */
    fun verifyFace(testEmbedding: FloatArray, storedEmbeddings: List<FloatArray>): Float {
        if (storedEmbeddings.isEmpty()) {
            return 0f
        }

        // Find maximum similarity across all stored embeddings
        return storedEmbeddings.maxOf { storedEmbedding ->
            cosineSimilarity(testEmbedding, storedEmbedding)
        }
    }

    /**
     * Normalize embedding using L2 normalization
     */
    private fun normalizeEmbedding(embedding: FloatArray): FloatArray {
        var norm = 0f
        for (value in embedding) {
            norm += value * value
        }
        norm = sqrt(norm)

        if (norm > 0) {
            for (i in embedding.indices) {
                embedding[i] /= norm
            }
        }
        return embedding
    }

    /**
     * Copy asset file to cache directory
     */
    private fun assetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)
        
        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }

        context.assets.open(assetName).use { inputStream ->
            FileOutputStream(file).use { outputStream ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (inputStream.read(buffer).also { read = it } != -1) {
                    outputStream.write(buffer, 0, read)
                }
                outputStream.flush()
            }
        }
        return file.absolutePath
    }

    /**
     * Clean up resources
     */
    fun close() {
        try {
            model.destroy()
        } catch (e: Exception) {
            Log.e(TAG, "Error closing model: ${e.message}", e)
        }
    }
}
