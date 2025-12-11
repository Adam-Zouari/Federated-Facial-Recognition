package com.faceid

import android.os.Bundle
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import com.faceid.databinding.ActivityDatabaseBinding
import com.faceid.data.FaceDatabase
import com.faceid.data.FaceEmbedding

class DatabaseActivity : AppCompatActivity() {
    
    private lateinit var binding: ActivityDatabaseBinding
    private lateinit var faceDatabase: FaceDatabase
    private lateinit var adapter: FaceListAdapter
    private var deleteMode = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityDatabaseBinding.inflate(layoutInflater)
        setContentView(binding.root)

        faceDatabase = FaceDatabase.getInstance(this)
        deleteMode = intent.getBooleanExtra("DELETE_MODE", false)

        setupRecyclerView()
        loadFaces()
    }

    private fun setupRecyclerView() {
        adapter = FaceListAdapter(
            onItemClick = { face ->
                if (deleteMode) {
                    confirmDelete(face)
                } else {
                    showFaceDetails(face)
                }
            }
        )
        
        binding.recyclerView.layoutManager = LinearLayoutManager(this)
        binding.recyclerView.adapter = adapter

        // Update title
        title = if (deleteMode) "Delete Face" else "Face Database"
    }

    private fun loadFaces() {
        val faces = faceDatabase.getAllFaces()
        
        if (faces.isEmpty()) {
            binding.tvEmptyMessage.visibility = android.view.View.VISIBLE
            binding.recyclerView.visibility = android.view.View.GONE
            binding.tvEmptyMessage.text = "No faces registered yet"
        } else {
            binding.tvEmptyMessage.visibility = android.view.View.GONE
            binding.recyclerView.visibility = android.view.View.VISIBLE
            adapter.submitList(faces)
        }
    }

    private fun showFaceDetails(face: FaceEmbedding) {
        AlertDialog.Builder(this)
            .setTitle("Face Details")
            .setMessage(
                "Name: ${face.name}\n\n" +
                "Poses: ${face.numPoses}\n\n" +
                "Date Added: ${face.dateAdded}"
            )
            .setPositiveButton("OK", null)
            .show()
    }

    private fun confirmDelete(face: FaceEmbedding) {
        AlertDialog.Builder(this)
            .setTitle("Confirm Delete")
            .setMessage("Are you sure you want to delete '${face.name}'?")
            .setPositiveButton("Delete") { _, _ ->
                deleteFace(face)
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun deleteFace(face: FaceEmbedding) {
        val success = faceDatabase.deleteFace(face.name)
        if (success) {
            Toast.makeText(
                this,
                "'${face.name}' has been deleted",
                Toast.LENGTH_SHORT
            ).show()
            loadFaces()
            
            // If no more faces, finish activity
            if (faceDatabase.isEmpty()) {
                finish()
            }
        } else {
            Toast.makeText(
                this,
                "Failed to delete face",
                Toast.LENGTH_SHORT
            ).show()
        }
    }
}
