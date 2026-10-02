package com.euhack.hello

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Color
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
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
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.FileOutputOptions
import androidx.camera.video.FallbackStrategy
import androidx.camera.video.Quality
import androidx.camera.video.QualitySelector
import androidx.camera.video.Recorder
import androidx.camera.video.Recording
import androidx.camera.video.VideoCapture
import androidx.camera.video.VideoRecordEvent
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Dataset capture: back-camera preview, Start/Stop records a silent 1080p video.
 * Videos land in <external files>/videos/ — pull them with adb and extract frames on the desktop.
 */
class MainActivity : ComponentActivity() {
    private lateinit var previewView: PreviewView
    private lateinit var elapsed: TextView
    private lateinit var status: TextView
    private lateinit var toggle: Button

    private var videoCapture: VideoCapture<Recorder>? = null
    private var recording: Recording? = null
    private var recorded = 0

    private val requestCamera =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera() else status.text = "Camera permission denied"
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
        stopRecording()
    }

    private fun buildLayout(): View {
        previewView = PreviewView(this).apply { scaleType = PreviewView.ScaleType.FIT_CENTER }
        elapsed = TextView(this).apply {
            textSize = 32f
            setTextColor(Color.RED)
            setShadowLayer(8f, 0f, 0f, Color.BLACK)
            gravity = Gravity.CENTER_HORIZONTAL
            setPadding(0, 160, 0, 0)
            visibility = View.INVISIBLE
        }
        status = TextView(this).apply {
            textSize = 16f
            setTextColor(Color.WHITE)
            setShadowLayer(6f, 0f, 0f, Color.BLACK)
            text = "Ready"
        }
        toggle = Button(this).apply {
            text = "Record"
            setOnClickListener { if (recording != null) stopRecording() else startRecording() }
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
            addView(elapsed, FrameLayout.LayoutParams(-1, -2, Gravity.TOP))
            addView(bottom, FrameLayout.LayoutParams(-1, -2, Gravity.BOTTOM))
        }
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()
            val preview = Preview.Builder().build().also { it.surfaceProvider = previewView.surfaceProvider }
            val recorder = Recorder.Builder()
                .setQualitySelector(
                    QualitySelector.from(Quality.FHD, FallbackStrategy.higherQualityOrLowerThan(Quality.FHD))
                )
                .setTargetVideoEncodingBitRate(BITRATE)
                .build()
            val capture = VideoCapture.withOutput(recorder)
            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, capture)
            videoCapture = capture
            status.text = "Ready — press Record"
        }, ContextCompat.getMainExecutor(this))
    }

    private fun startRecording() {
        val capture = videoCapture ?: return
        val dir = File(getExternalFilesDir(null), "videos").apply { mkdirs() }
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val file = File(dir, "hand_$stamp.mp4")
        recording = capture.output
            .prepareRecording(this, FileOutputOptions.Builder(file).build())
            .start(ContextCompat.getMainExecutor(this)) { event -> onRecordEvent(event, file) }
        toggle.text = "Stop"
        elapsed.visibility = View.VISIBLE
    }

    private fun stopRecording() {
        recording?.stop()
        recording = null
        toggle.text = "Record"
        elapsed.visibility = View.INVISIBLE
    }

    private fun onRecordEvent(event: VideoRecordEvent, file: File) {
        when (event) {
            is VideoRecordEvent.Status -> {
                val s = event.recordingStats.recordedDurationNanos / 1_000_000_000
                elapsed.text = String.format(Locale.US, "● %d:%02d", s / 60, s % 60)
            }
            is VideoRecordEvent.Finalize -> {
                if (event.hasError()) {
                    Log.e(TAG, "recording error ${event.error}", event.cause)
                    status.text = "Recording error ${event.error}: ${event.cause?.message}"
                } else {
                    recorded += 1
                    status.text = "Saved ${file.name} (${file.length() / 1_000_000} MB) — $recorded this run"
                }
            }
            else -> Unit
        }
    }

    companion object {
        private const val TAG = "HandCapture"
        private const val BITRATE = 20_000_000  // high bitrate keeps finger edges free of compression mush
    }
}
