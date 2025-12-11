package com.faceid

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.faceid.databinding.ActivityMainBinding
import com.faceid.data.FaceDatabase

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private lateinit var faceDatabase: FaceDatabase

    companion object {
        private const val CAMERA_PERMISSION_REQUEST_CODE = 100
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Initialize database
        faceDatabase = FaceDatabase.getInstance(this)

        setupUI()
        checkCameraPermission()
    }

    private fun setupUI() {
        // Update database count
        updateDatabaseCount()

        // Add New Face button
        binding.btnAddFace.setOnClickListener {
            try {
                if (hasCameraPermission()) {
                    startActivity(Intent(this, EnrollmentActivity::class.java))
                } else {
                    requestCameraPermission()
                }
            } catch (e: Exception) {
                Toast.makeText(
                    this,
                    "Error: ${e.message}",
                    Toast.LENGTH_LONG
                ).show()
                e.printStackTrace()
            }
        }

        // Verify Face button
        binding.btnVerifyFace.setOnClickListener {
            if (faceDatabase.isEmpty()) {
                Toast.makeText(
                    this,
                    "No faces in database. Please add a face first.",
                    Toast.LENGTH_LONG
                ).show()
                return@setOnClickListener
            }

            if (hasCameraPermission()) {
                startActivity(Intent(this, VerificationActivity::class.java))
            } else {
                requestCameraPermission()
            }
        }

        // View Database button
        binding.btnViewDatabase.setOnClickListener {
            startActivity(Intent(this, DatabaseActivity::class.java))
        }

        // Delete Face button
        binding.btnDeleteFace.setOnClickListener {
            if (faceDatabase.isEmpty()) {
                Toast.makeText(this, "No faces in database.", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            startActivity(Intent(this, DatabaseActivity::class.java).apply {
                putExtra("DELETE_MODE", true)
            })
        }
    }

    private fun updateDatabaseCount() {
        val count = faceDatabase.getAllFaces().size
        binding.tvDatabaseCount.text = "Database: $count registered face${if (count != 1) "s" else ""}"
    }

    override fun onResume() {
        super.onResume()
        updateDatabaseCount()
    }

    private fun hasCameraPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun requestCameraPermission() {
        ActivityCompat.requestPermissions(
            this,
            arrayOf(Manifest.permission.CAMERA),
            CAMERA_PERMISSION_REQUEST_CODE
        )
    }

    private fun checkCameraPermission() {
        if (!hasCameraPermission()) {
            requestCameraPermission()
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        when (requestCode) {
            CAMERA_PERMISSION_REQUEST_CODE -> {
                if (grantResults.isNotEmpty() && 
                    grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    Toast.makeText(this, "Camera permission granted", Toast.LENGTH_SHORT).show()
                } else {
                    Toast.makeText(
                        this,
                        "Camera permission is required for face recognition",
                        Toast.LENGTH_LONG
                    ).show()
                }
            }
        }
    }
}
