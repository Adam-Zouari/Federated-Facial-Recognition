package com.faceid.data

import android.content.Context
import android.util.Log
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import java.io.File
import java.text.SimpleDateFormat
import java.util.*

/**
 * Data class representing a stored face with embeddings
 */
data class FaceEmbedding(
    val name: String,
    val embeddings: List<FloatArray>,
    val dateAdded: String,
    val numPoses: Int = embeddings.size
)

/**
 * Singleton database for managing face embeddings
 */
class FaceDatabase private constructor(context: Context) {
    
    private val gson = Gson()
    private val databaseFile = File(context.filesDir, DATABASE_FILE)
    private val faces = mutableMapOf<String, FaceEmbedding>()
    
    companion object {
        private const val TAG = "FaceDatabase"
        private const val DATABASE_FILE = "face_database.json"
        
        @Volatile
        private var instance: FaceDatabase? = null
        
        fun getInstance(context: Context): FaceDatabase {
            return instance ?: synchronized(this) {
                instance ?: FaceDatabase(context.applicationContext).also { instance = it }
            }
        }
    }

    init {
        loadDatabase()
    }

    /**
     * Load database from file
     */
    private fun loadDatabase() {
        if (!databaseFile.exists()) {
            Log.d(TAG, "Database file not found, starting with empty database")
            return
        }

        try {
            val json = databaseFile.readText()
            val type = object : TypeToken<Map<String, SerializableFace>>() {}.type
            val loadedFaces: Map<String, SerializableFace> = gson.fromJson(json, type)
            
            faces.clear()
            loadedFaces.forEach { (name, serializableFace) ->
                faces[name] = FaceEmbedding(
                    name = name,
                    embeddings = serializableFace.embeddings,
                    dateAdded = serializableFace.dateAdded,
                    numPoses = serializableFace.embeddings.size
                )
            }
            
            Log.d(TAG, "Loaded ${faces.size} faces from database")
        } catch (e: Exception) {
            Log.e(TAG, "Error loading database: ${e.message}", e)
            faces.clear()
        }
    }

    /**
     * Save database to file
     */
    @Synchronized
    private fun saveDatabase() {
        try {
            val serializableFaces = faces.mapValues { (_, face) ->
                SerializableFace(
                    embeddings = face.embeddings,
                    dateAdded = face.dateAdded
                )
            }
            
            val json = gson.toJson(serializableFaces)
            databaseFile.writeText(json)
            
            Log.d(TAG, "Database saved successfully (${faces.size} faces)")
        } catch (e: Exception) {
            Log.e(TAG, "Error saving database: ${e.message}", e)
        }
    }

    /**
     * Add or update a face in the database
     */
    @Synchronized
    fun addFace(name: String, embeddings: List<FloatArray>): Boolean {
        if (name.isBlank() || embeddings.isEmpty()) {
            Log.e(TAG, "Invalid name or empty embeddings")
            return false
        }

        val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault())
        val faceEmbedding = FaceEmbedding(
            name = name,
            embeddings = embeddings,
            dateAdded = dateFormat.format(Date())
        )

        faces[name] = faceEmbedding
        saveDatabase()
        
        Log.d(TAG, "Added/Updated face: $name with ${embeddings.size} poses")
        return true
    }

    /**
     * Get a face by name
     */
    fun getFace(name: String): FaceEmbedding? {
        return faces[name]
    }

    /**
     * Get all faces
     */
    fun getAllFaces(): List<FaceEmbedding> {
        return faces.values.toList()
    }

    /**
     * Delete a face by name
     */
    @Synchronized
    fun deleteFace(name: String): Boolean {
        val removed = faces.remove(name)
        if (removed != null) {
            saveDatabase()
            Log.d(TAG, "Deleted face: $name")
            return true
        }
        Log.w(TAG, "Face not found: $name")
        return false
    }

    /**
     * Check if database is empty
     */
    fun isEmpty(): Boolean {
        return faces.isEmpty()
    }

    /**
     * Check if a name exists
     */
    fun contains(name: String): Boolean {
        return faces.containsKey(name)
    }

    /**
     * Clear all faces
     */
    @Synchronized
    fun clear() {
        faces.clear()
        saveDatabase()
        Log.d(TAG, "Database cleared")
    }

    /**
     * Serializable version of FaceEmbedding for JSON
     */
    private data class SerializableFace(
        val embeddings: List<FloatArray>,
        val dateAdded: String
    )
}
