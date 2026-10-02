package com.euhack.hello

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Color
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.util.Size
import android.view.Gravity
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Dataset capture: back-camera preview, one photo every [INTERVAL_S] seconds while running.
 * Photos land in <external files>/captures/<session>/ — pull them with adb (see README).
 */
class MainActivity : ComponentActivity() {
    private lateinit var previewView: PreviewView
    private lateinit var countdown: TextView
    private lateinit var status: TextView
    private lateinit var toggle: Button
    private lateinit var flash: View

    private var imageCapture: ImageCapture? = null
    private val handler = Handler(Looper.getMainLooper())
    private var running = false
    private var secondsLeft = INTERVAL_S
    private var sessionDir: File? = null
    private var captured = 0

    private val requestCamera =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera() else status.text = "Camera permission denied"
        }

    private val tick = object : Runnable {
        override fun run() {
            if (!running) return
            secondsLeft -= 1
            if (secondsLeft <= 0) {
                takePhoto()
                secondsLeft = INTERVAL_S
            }
            countdown.text = secondsLeft.toString()
            handler.postDelayed(this, 1000)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContentView(buildLayout())

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            requestCamera.launch(Manifest.permission.CAMERA)
        }
    }

    override fun onPause() {
        super.onPause()
        stopCapture()
    }

    private fun buildLayout(): View {
        previewView = PreviewView(this).apply { scaleType = PreviewView.ScaleType.FIT_CENTER }
        flash = View(this).apply {
            setBackgroundColor(Color.WHITE)
            alpha = 0f
        }
        countdown = TextView(this).apply {
            textSize = 96f
            setTextColor(Color.WHITE)
            setShadowLayer(8f, 0f, 0f, Color.BLACK)
            gravity = Gravity.CENTER
            visibility = View.INVISIBLE
        }
        status = TextView(this).apply {
            textSize = 16f
            setTextColor(Color.WHITE)
            setShadowLayer(6f, 0f, 0f, Color.BLACK)
            text = "Ready"
        }
        toggle = Button(this).apply {
            text = "Start"
            setOnClickListener { if (running) stopCapture() else startCapture() }
        }
        val bottom = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER_HORIZONTAL
            // Android 15 draws edge-to-edge: keep the controls above the navigation bar.
            setOnApplyWindowInsetsListener { v, insets ->
                v.setPadding(32, 32, 32, 32 + insets.systemWindowInsetBottom)
                insets
            }
            addView(status)
            addView(toggle)
        }
        return FrameLayout(this).apply {
            setBackgroundColor(Color.BLACK)
            addView(previewView, FrameLayout.LayoutParams(-1, -1))
            addView(countdown, FrameLayout.LayoutParams(-1, -1))
            addView(flash, FrameLayout.LayoutParams(-1, -1))
            addView(bottom, FrameLayout.LayoutParams(-1, -2, Gravity.BOTTOM))
        }
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()
            val resolution = ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                .setResolutionStrategy(
                    ResolutionStrategy(TARGET_SIZE, ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)
                )
                .build()
            val preview = Preview.Builder().build().also { it.surfaceProvider = previewView.surfaceProvider }
            val capture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .setResolutionSelector(resolution)
                .build()
            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, capture)
            imageCapture = capture
            status.text = "Ready — press Start"
        }, ContextCompat.getMainExecutor(this))
    }

    private fun startCapture() {
        if (imageCapture == null) return
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        sessionDir = File(getExternalFilesDir(null), "captures/session_$stamp").apply { mkdirs() }
        captured = 0
        running = true
        secondsLeft = INTERVAL_S
        toggle.text = "Stop"
        countdown.text = secondsLeft.toString()
        countdown.visibility = View.VISIBLE
        updateStatus()
        handler.postDelayed(tick, 1000)
    }

    private fun stopCapture() {
        running = false
        handler.removeCallbacks(tick)
        countdown.visibility = View.INVISIBLE
        toggle.text = "Start"
        if (sessionDir != null) updateStatus()
    }

    private fun takePhoto() {
        val capture = imageCapture ?: return
        val dir = sessionDir ?: return
        val name = SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US).format(Date())
        val file = File(dir, "hand_$name.jpg")
        capture.takePicture(
            ImageCapture.OutputFileOptions.Builder(file).build(),
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                    captured += 1
                    flash.alpha = 0.6f
                    flash.animate().alpha(0f).setDuration(250).start()
                    updateStatus()
                }

                override fun onError(e: ImageCaptureException) {
                    Log.e(TAG, "capture failed", e)
                    status.text = "Capture failed: ${e.message}"
                }
            },
        )
    }

    private fun updateStatus() {
        status.text = "${if (running) "Capturing" else "Stopped"} — $captured photos\n${sessionDir?.name}"
    }

    companion object {
        private const val TAG = "HandCapture"
        private const val INTERVAL_S = 5
        private val TARGET_SIZE = Size(1920, 1440)
    }
}
