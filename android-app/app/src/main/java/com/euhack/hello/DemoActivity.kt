package com.euhack.hello

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Typeface
import android.hardware.camera2.CameraCharacteristics
import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.view.Gravity
import android.view.View
import android.view.WindowManager
import android.app.AlertDialog
import android.widget.ProgressBar
import android.widget.Button
import android.widget.FrameLayout
import android.widget.PopupMenu
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.OptIn
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.FallbackStrategy
import androidx.camera.video.FileOutputOptions
import androidx.camera.video.Quality
import androidx.camera.video.QualitySelector
import androidx.camera.video.Recorder
import androidx.camera.video.Recording
import androidx.camera.video.VideoCapture
import androidx.camera.video.VideoRecordEvent
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.json.JSONArray
import org.json.JSONObject
import org.pytorch.executorch.Tensor
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.sqrt

/**
 * Live hand demo: the centred square of the camera frame goes through HandSegNet (ExecuTorch, XNNPACK),
 * and the predicted outline is drawn in green over the preview when the model says a hand is present.
 *
 * The Options dropdown toggles:
 *  - Record: silent 1080p dataset video to <external files>/videos/ (see ML/README.md), plus a
 *    [SessionRecorder] session of every analysed camera frame and every thermal packet, time-stamped
 *    on one clock, for fitting the camera-to-thermal alignment;
 *  - Dump NO-HAND frames: [NoHandLogger] diagnostics;
 *  - Stream thermal input: shows only the USB thermal camera ([ThermalUsbStream]) full screen. The camera
 *    keeps running behind it so recording still captures both;
 *  - Calibrate thermal ↔ camera: 5 s of the hand seen by both cameras -> [ThermalCalibration] finds where
 *    the thermal camera sits (angles, offsets), shown in a popup and saved ([CalibrationStore]);
 *  - Fused thermal view (once calibrated): only the part of the camera image the thermal camera also sees,
 *    with temperatures upsampled along the camera's edges ([FusionRenderer]); the rest is black.
 * Whenever the thermal camera is plugged in (and not fused), a small live thermal view sits in the corner.
 */
class DemoActivity : ComponentActivity() {
    private lateinit var previewView: PreviewView
    private lateinit var outlineView: OutlineOverlay
    private lateinit var status: TextView

    private val analysisExecutor = Executors.newSingleThreadExecutor()
    private var module: Module? = null
    private val input = FloatArray(3 * SIZE * SIZE)
    private val pixels = IntArray(SIZE * SIZE)
    private var lastFrameMs = 0L
    private lateinit var logger: NoHandLogger
    private lateinit var thermalView: ThermalView
    private lateinit var thermalInset: ThermalView
    private lateinit var sessions: SessionRecorder
    private lateinit var recordingLabel: TextView

    private var cameraProvider: ProcessCameraProvider? = null
    private var camera: Camera? = null
    private var videoCapture: VideoCapture<Recorder>? = null
    private var recording: Recording? = null
    private var recorded = 0
    private var recordStartMs = 0L
    private var thermal: ThermalUsbStream? = null
    private var thermalMode = false
    @Volatile private var thermalStatus = "Starting thermal input…"
    @Volatile private var latestThermal: ThermalFrame? = null

    private lateinit var fusedView: FusedView
    private lateinit var calibProgress: ProgressBar
    private lateinit var calibStore: CalibrationStore
    @Volatile private var calibration: ThermalCalibration.Result? = null
    @Volatile private var fusedMode = true
    @Volatile private var fusedVisible = false
    @Volatile private var fusedStatus = ""
    private val fusion = FusionRenderer()
    private val fusionExecutor = Executors.newSingleThreadExecutor()
    private val fusionBusy = AtomicBoolean(false)
    /** Phone camera focal length in sensor (active array) pixels, from Camera2; 0 until bound. */
    @Volatile private var sensorFocalPx = 0.0

    private enum class CalibState { OFF, COLLECTING, SOLVING }
    @Volatile private var calibState = CalibState.OFF
    private val calibCams = ArrayList<ThermalCalibration.CameraSample>()      // analysis thread only
    private val calibThermals = ConcurrentLinkedQueue<ThermalCalibration.ThermalSample>()
    @Volatile private var collectThermal = false
    private var calibHandMs = 0L
    @Volatile private var calibNeedsDepth = false
    private var calibLastHandMs = 0L
    private var lastThermalMs = 0L

    private val requestCamera =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera() else status.text = "Camera permission denied"
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        previewView = PreviewView(this).apply {
            scaleType = PreviewView.ScaleType.FIT_CENTER
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        }
        outlineView = OutlineOverlay(this)
        status = TextView(this).apply {
            textSize = 18f
            setTextColor(Color.WHITE)
            setBackgroundColor(0x99000000.toInt())
            setPadding(24, 16, 24, 16)
            gravity = Gravity.CENTER
            text = "Loading model…"
        }
        logger = NoHandLogger(getExternalFilesDir(null)!!)
        sessions = SessionRecorder(getExternalFilesDir(null)!!)
        thermalView = ThermalView(this).apply { visibility = View.GONE }
        thermalInset = ThermalView(this).apply { visibility = View.GONE }
        fusedView = FusedView(this, fusion).apply { visibility = View.GONE }
        calibProgress = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply {
            max = 1000
            visibility = View.GONE
        }
        calibStore = CalibrationStore(filesDir)
        calibration = calibStore.load()
        recordingLabel = TextView(this).apply {
            textSize = 28f
            setTextColor(Color.RED)
            setShadowLayer(8f, 0f, 0f, Color.BLACK)
            gravity = Gravity.CENTER_HORIZONTAL
            visibility = View.INVISIBLE
        }
        val options = Button(this).apply {
            text = "Options ▾"
            setOnClickListener { showOptions(it) }
        }
        setContentView(FrameLayout(this).apply {
            setBackgroundColor(Color.BLACK)
            addView(previewView, FrameLayout.LayoutParams(-1, -1))
            addView(fusedView, FrameLayout.LayoutParams(-1, -1))
            addView(outlineView, FrameLayout.LayoutParams(-1, -1))
            addView(thermalView, FrameLayout.LayoutParams(-1, -1))
            addView(thermalInset, FrameLayout.LayoutParams(INSET_W, INSET_W * 3 / 4, Gravity.TOP or Gravity.END)
                .apply { topMargin = 330; rightMargin = 24 })
            addView(status, FrameLayout.LayoutParams(-1, -2, Gravity.TOP).apply { topMargin = 120 })
            addView(calibProgress, FrameLayout.LayoutParams(-1, -2, Gravity.TOP)
                .apply { topMargin = 300; leftMargin = 48; rightMargin = 48 })
            addView(recordingLabel, FrameLayout.LayoutParams(-1, -2, Gravity.TOP).apply { topMargin = 360 })
            addView(options, FrameLayout.LayoutParams(-2, -2, Gravity.BOTTOM or Gravity.CENTER_HORIZONTAL)
                .apply { bottomMargin = 200 })
        })

        analysisExecutor.execute {
            try {
                module = Module.load(assetFilePath(this, MODEL_ASSET))
                runOnUiThread { status.text = "Model loaded — hold your hand in the box" }
            } catch (e: Exception) {
                Log.e(TAG, "model load failed", e)
                runOnUiThread { status.text = "Model load failed: ${e.message}" }
            }
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            requestCamera.launch(Manifest.permission.CAMERA)
        }
    }

    private fun showOptions(anchor: View) {
        PopupMenu(this, anchor).apply {
            menu.add(0, MENU_RECORD, 0, "Record camera + thermal").apply {
                isCheckable = true
                isChecked = sessions.active
            }
            menu.add(0, MENU_DUMP, 1, "Dump NO-HAND frames").apply {
                isCheckable = true
                isChecked = logger.enabled
            }
            menu.add(0, MENU_THERMAL, 2, "Stream thermal input").apply {
                isCheckable = true
                isChecked = thermalMode
            }
            menu.add(0, MENU_CALIBRATE, 3, "Calibrate thermal ↔ camera").apply {
                isEnabled = calibState == CalibState.OFF
            }
            menu.add(0, MENU_FUSED, 4, "Fused thermal view").apply {
                isCheckable = true
                isChecked = fusedMode && calibration != null
                isEnabled = calibration != null
            }
            setOnMenuItemClickListener { item ->
                when (item.itemId) {
                    MENU_RECORD -> if (sessions.active) stopRecording() else startRecording()
                    MENU_DUMP -> setLogging(!logger.enabled)
                    MENU_THERMAL -> setThermalMode(!thermalMode)
                    MENU_CALIBRATE -> startCalibration()
                    MENU_FUSED -> fusedMode = !fusedMode
                }
                true
            }
        }.show()
    }

    private fun setLogging(on: Boolean) {
        // Flip on the analysis thread so a frame never sees a half-started session.
        analysisExecutor.execute {
            if (on == logger.enabled) return@execute
            if (on) {
                Log.i(TAG, "diagnostics -> ${logger.start()}")
            } else {
                logger.stop()
                val saved = logger.dumpCount
                runOnUiThread { status.text = "NO-HAND logging off — $saved frames saved" }
            }
        }
    }

    /** Starts the session (camera frames + thermal packets) and, if the camera supports it, the video. */
    private fun startRecording() {
        if (sessions.active) return
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val meta = sessionMeta(stamp)
        videoCapture?.let { capture ->
            val dir = File(getExternalFilesDir(null), "videos").apply { mkdirs() }
            val file = File(dir, "hand_$stamp.mp4")
            meta.put("video", "videos/${file.name}")
            recording = capture.output
                .prepareRecording(this, FileOutputOptions.Builder(file).build())
                .start(ContextCompat.getMainExecutor(this)) { event -> onRecordEvent(event, file) }
        }
        val dir = sessions.start(stamp, meta)
        Log.i(TAG, "session -> $dir")
        recordStartMs = SystemClock.elapsedRealtime()
        recordingLabel.text = "● 0:00"
        recordingLabel.visibility = View.VISIBLE
    }

    private fun stopRecording() {
        if (!sessions.active && recording == null) return
        val frames = sessions.cameraFrames
        val packets = sessions.thermalPackets
        sessions.stop()
        recording?.stop()
        recording = null
        recordingLabel.text = "Saved session: $frames camera frames, $packets thermal packets"
    }

    /** Called on the UI thread while recording, from the analysis loop. */
    private fun updateRecordingLabel() {
        if (!sessions.active) return
        val s = (SystemClock.elapsedRealtime() - recordStartMs) / 1000
        recordingLabel.text = String.format(Locale.US, "● %d:%02d   %d cam   %d thermal",
            s / 60, s % 60, sessions.cameraFrames, sessions.thermalPackets)
    }

    private fun onRecordEvent(event: VideoRecordEvent, file: File) {
        if (event is VideoRecordEvent.Finalize) {
            if (event.hasError()) {
                Log.e(TAG, "recording error ${event.error}", event.cause)
                recordingLabel.text = "Video error ${event.error}: ${event.cause?.message}"
            } else {
                recorded += 1
                Log.i(TAG, "saved ${file.name} (${file.length() / 1_000_000} MB), $recorded this run")
            }
        }
    }

    /** Everything needed later to map camera pixels to rays: camera characteristics and device info. */
    @OptIn(markerClass = [ExperimentalCamera2Interop::class])
    private fun sessionMeta(stamp: String): JSONObject {
        val meta = JSONObject()
            .put("created", stamp)
            .put("device", "${Build.MANUFACTURER} ${Build.MODEL}")
            .put("model_asset", MODEL_ASSET)
            .put("thermal_sensor", "MLX90640-BAB (Adafruit 4407) 55x35 deg, taped to the back of the phone")
        val info = camera?.cameraInfo ?: return meta
        val c2 = Camera2CameraInfo.from(info)
        fun floats(values: FloatArray?) = values?.let { JSONArray(it.map { v -> v.toDouble() }) }
        val cam = JSONObject().put("camera_id", c2.cameraId)
        cam.putOpt("intrinsic_calibration", floats(c2.getCameraCharacteristic(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION)))
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            cam.putOpt("distortion", floats(c2.getCameraCharacteristic(CameraCharacteristics.LENS_DISTORTION)))
        }
        cam.putOpt("pose_translation", floats(c2.getCameraCharacteristic(CameraCharacteristics.LENS_POSE_TRANSLATION)))
        cam.putOpt("pose_rotation", floats(c2.getCameraCharacteristic(CameraCharacteristics.LENS_POSE_ROTATION)))
        cam.putOpt("focal_lengths_mm", floats(c2.getCameraCharacteristic(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)))
        c2.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)?.let {
            cam.put("sensor_physical_mm", JSONArray(listOf(it.width.toDouble(), it.height.toDouble())))
        }
        c2.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE)?.let {
            cam.put("pixel_array", JSONArray(listOf(it.width, it.height)))
        }
        c2.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)?.let {
            cam.put("active_array", JSONArray(listOf(it.left, it.top, it.right, it.bottom)))
        }
        cam.putOpt("sensor_orientation", c2.getCameraCharacteristic(CameraCharacteristics.SENSOR_ORIENTATION))
        cam.put("timestamp_source",
            if (c2.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE) ==
                CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME) "realtime" else "unknown")
        return meta.put("camera", cam)
    }

    // ------------------------------------------------------------------------------------ calibration

    private fun thermalLive(): Boolean =
        latestThermal?.let { SystemClock.elapsedRealtimeNanos() - it.receivedNs < 1_500_000_000L } ?: false

    private fun startCalibration() {
        if (calibState != CalibState.OFF) return
        if (!thermalLive()) {
            status.text = "Plug in the thermal camera first"
            return
        }
        setThermalMode(false)
        analysisExecutor.execute {
            calibCams.clear()
            calibHandMs = 0L
            calibLastHandMs = 0L
            calibNeedsDepth = false
        }
        calibThermals.clear()
        collectThermal = true
        calibState = CalibState.COLLECTING
        calibProgress.isIndeterminate = false
        calibProgress.progress = 0
        calibProgress.visibility = View.VISIBLE
        status.setTextColor(Color.WHITE)
        status.text = CALIBRATE_PROMPT
    }

    /** Analysis thread: keep frames where both cameras see the hand; 5 s of them completes the capture. */
    private fun collectCalibration(tsNs: Long, mask: FloatArray, left: Int, top: Int, side: Int,
                                   cam: CameraIntrinsics, handVisible: Boolean, nowMs: Long) {
        val sample = if (handVisible && thermalLive())
            ThermalCalibration.cameraSample(tsNs, mask, SIZE, left, top, side, cam) else null
        if (sample == null) {
            calibLastHandMs = 0L
            return
        }
        calibCams.add(sample)
        if (calibLastHandMs > 0) calibHandMs += minOf(nowMs - calibLastHandMs, 200L)
        calibLastHandMs = nowMs
        // Parallax separates tilt from offset, so the capture also needs the hand near and far: after 5 s it
        // keeps going until the depth spread is there (or 15 s have passed).
        val spread = ThermalCalibration.depthSpread(calibCams)
        val needDepth = calibHandMs >= CALIBRATION_MS && spread < MIN_DEPTH_SPREAD
        calibNeedsDepth = needDepth
        val progress = if (!needDepth) (calibHandMs * 1000 / CALIBRATION_MS).toInt().coerceAtMost(1000)
            else (900 + 100 * (spread - 1) / (MIN_DEPTH_SPREAD - 1)).toInt().coerceAtMost(999)
        runOnUiThread { calibProgress.progress = progress }
        if (calibHandMs >= CALIBRATION_MS && (!needDepth || calibHandMs >= MAX_CALIBRATION_MS)) {
            calibState = CalibState.SOLVING
            val cams = ArrayList(calibCams)
            Thread({ solveCalibration(cams) }, "calibration").start()
        }
    }

    private fun solveCalibration(cams: List<ThermalCalibration.CameraSample>) {
        runOnUiThread { status.text = "Calibrating — solving for the thermal camera's position…" }
        // Thermal packets arrive a few hundred ms after the matching camera frame: wait for the last ones.
        SystemClock.sleep(600)
        collectThermal = false
        val thermals = calibThermals.toList()
        val camera = cams.firstOrNull()?.let { intrinsicsForCalibration } ?: return finishCalibration(null)
        val result = try {
            ThermalCalibration.solve(cams, thermals, camera) { p ->
                runOnUiThread { calibProgress.progress = (p * 1000).toInt() }
            }
        } catch (e: Exception) {
            Log.e(TAG, "calibration failed", e)
            null
        }
        finishCalibration(result)
    }

    @Volatile private var intrinsicsForCalibration: CameraIntrinsics? = null

    private fun finishCalibration(result: ThermalCalibration.Result?) = runOnUiThread {
        calibState = CalibState.OFF
        calibProgress.visibility = View.GONE
        val dialog = AlertDialog.Builder(this)
        if (result == null) {
            dialog.setTitle("Not enough data")
                .setMessage("Both cameras need to see your hand for 5 seconds. Keep it inside the box, " +
                    "in front of the thermal camera, and move it around slowly.")
                .setPositiveButton("Retry") { _, _ -> startCalibration() }
                .setNegativeButton("Cancel", null)
        } else {
            val good = result.correlation >= 0.6
            dialog.setTitle(if (good) "Calibrated" else "Calibration looks poor")
                .setMessage(CalibrationStore.describe(result) + if (good) "" else
                    "\n\nTry again with your hand filling more of the box, moving nearer and further.")
                .setPositiveButton("Use it") { _, _ ->
                    calibStore.save(result)
                    calibration = result
                    fusedMode = true
                }
                .setNeutralButton("Retry") { _, _ -> startCalibration() }
                .setNegativeButton("Cancel", null)
        }
        dialog.show().findViewById<TextView>(android.R.id.message)?.apply {
            typeface = Typeface.MONOSPACE
            textSize = 13f
        }
    }

    /** Phone camera intrinsics for the upright frame, from Camera2 and CameraX's sensor-to-buffer transform. */
    private fun intrinsics(sensorToBuffer: FloatArray, frameW: Int, frameH: Int, bufferW: Int): CameraIntrinsics {
        val f = if (sensorFocalPx > 0) sensorFocalPx * sensorToBuffer[0]
            else bufferW / 2 / Math.tan(Math.toRadians(65.0 / 2))      // typical phone main camera
        return CameraIntrinsics(f, frameW / 2.0, frameH / 2.0)
    }

    @OptIn(markerClass = [ExperimentalCamera2Interop::class])
    private fun readFocalLength() {
        val info = camera?.cameraInfo ?: return
        val c2 = Camera2CameraInfo.from(info)
        val focal = c2.getCameraCharacteristic(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.firstOrNull()
        val size = c2.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
        val pixels = c2.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE)
        if (focal != null && size != null && pixels != null) sensorFocalPx = focal.toDouble() / size.width * pixels.width
    }

    /**
     * Thermal veto for the hand model's false positives (cluttered or blurred scenes it calls "hand"): a hand
     * is warm. Calibrated: the thermal pixels under the mask must be >= [VETO_DELTA_C] above the scene's 30th
     * percentile (on the recorded session this rejected 20 of 22 confirmed false positives and kept ~94% of
     * real hands). Uncalibrated or calibrating: something in the thermal view must be warm at all. Without a
     * live thermal camera, or if the hand is outside the thermal view, the model's answer stands.
     */
    private fun thermalAgrees(mask: FloatArray, left: Int, top: Int, side: Int, cam: CameraIntrinsics): Boolean {
        if (!thermalLive()) return true
        val t = latestThermal?.celsius ?: return true
        val cal = calibration
        if (cal == null || calibState != CalibState.OFF) return ThermalCalibration.warmth(t) != null
        var above = 0
        for (p in mask) if (p > 0.5f) above++
        val areaPx = above.toDouble() / mask.size * side * side
        if (areaPx < 500) return true
        val z = cam.f * sqrt(ThermalCalibration.HAND_AREA_CM2 / areaPx)
        val uv = DoubleArray(2)
        val temps = ArrayList<Float>()
        for (gy in 0 until VETO_GRID) for (gx in 0 until VETO_GRID) {
            val mx = ((gx + 0.5) / VETO_GRID * SIZE).toInt()
            val my = ((gy + 0.5) / VETO_GRID * SIZE).toInt()
            if (mask[my * SIZE + mx] <= 0.5f) continue
            val fx = left + (gx + 0.5) / VETO_GRID * side
            val fy = top + (gy + 0.5) / VETO_GRID * side
            cal.pose.project((fx - cam.cx) / cam.f * z, (fy - cam.cy) / cam.f * z, z, uv)
            val u = Math.round(uv[0]).toInt()
            val v = Math.round(uv[1]).toInt()
            if (u in 0 until ThermalGeometry.W && v in 0 until ThermalGeometry.H) temps.add(t[v * ThermalGeometry.W + u])
        }
        if (temps.size < 3) return true
        val sorted = t.sortedArray()
        val scene = sorted[(sorted.size * 0.3f).toInt()]
        return temps.sorted()[temps.size / 2] - scene >= VETO_DELTA_C
    }

    // ------------------------------------------------------------------------------------ fusion

    /** Analysis thread: hand the frame to the fusion thread unless it is still busy with the last one. */
    private fun renderFused(frame: Bitmap, mask: FloatArray, handVisible: Boolean, left: Int, top: Int, side: Int,
                            cam: CameraIntrinsics) {
        val cal = calibration ?: return
        val thermalFrame = latestThermal ?: return
        if (!fusionBusy.compareAndSet(false, true)) return
        var depth = FusionRenderer.DEFAULT_DEPTH_CM
        var hand: FloatArray? = null
        if (handVisible) {
            var above = 0
            for (p in mask) if (p > 0.5f) above++
            val areaPx = above.toDouble() / mask.size * side * side
            if (areaPx > 500) {
                depth = cam.f * sqrt(ThermalCalibration.HAND_AREA_CM2 / areaPx)
                hand = mask.copyOf()
            }
        }
        fusionExecutor.execute {
            try {
                val st = fusion.render(frame, thermalFrame, cal.pose, cam, depth, hand, SIZE, left, top, side)
                fusedView.update(frame.width, frame.height)
                fusedStatus = String.format(Locale.US, "%s%.1f–%.1f °C",
                    st.handC?.let { String.format(Locale.US, "HAND %.1f °C   |   ", it) } ?: "", st.lowC, st.highC)
            } finally {
                fusionBusy.set(false)
            }
        }
    }

    /** Draws the fused bitmap with the same FIT_CENTER placement as the preview and [OutlineOverlay]. */
    class FusedView(context: Context, private val fusion: FusionRenderer) : View(context) {
        private val paint = Paint().apply { isFilterBitmap = true }
        private val dst = RectF()
        @Volatile private var frameW = 0
        @Volatile private var frameH = 0

        fun update(w: Int, h: Int) {
            frameW = w
            frameH = h
            postInvalidate()
        }

        override fun onDraw(canvas: Canvas) {
            if (frameW == 0) return
            val scale = minOf(width.toFloat() / frameW, height.toFloat() / frameH)
            val w = frameW * scale
            val h = frameH * scale
            dst.set((width - w) / 2, (height - h) / 2, (width + w) / 2, (height + h) / 2)
            synchronized(fusion.bitmap) { canvas.drawBitmap(fusion.bitmap, null, dst, paint) }
        }
    }

    /** Thermal mode shows only the thermal stream; the camera keeps running (hidden) for recording. */
    private fun setThermalMode(on: Boolean) {
        if (on == thermalMode) return
        thermalMode = on
        val camera = if (on) View.GONE else View.VISIBLE
        previewView.visibility = camera
        outlineView.visibility = camera
        thermalView.visibility = if (on) View.VISIBLE else View.GONE
        thermalInset.visibility = View.GONE
        status.setTextColor(Color.WHITE)
        status.text = if (on) thermalStatus else "…"
    }

    private fun startThermal() {
        if (thermal != null) return
        lastThermalMs = 0L
        thermal = ThermalUsbStream(this, ::onThermalFrame) { text ->
            thermalStatus = text
            runOnUiThread {
                thermalInset.visibility = View.GONE
                if (thermalMode) status.text = text
            }
        }.also { it.start() }
    }

    private fun stopThermal() {
        thermal?.stop()
        thermal = null
    }

    /** Called on the USB thread. */
    private fun onThermalFrame(frame: ThermalFrame) {
        latestThermal = frame
        if (collectThermal) calibThermals.add(ThermalCalibration.ThermalSample(frame.receivedNs, frame.celsius))
        sessions.onThermalFrame(frame)
        if (!thermalMode && !fusedVisible) {
            thermalInset.update(frame)
            if (thermalInset.visibility != View.VISIBLE) runOnUiThread { if (!thermalMode) thermalInset.visibility = View.VISIBLE }
        }
        thermalView.update(frame)
        val nowMs = frame.receivedNs / 1_000_000
        val fps = if (lastThermalMs > 0) 1000f / (nowMs - lastThermalMs) else 0f
        lastThermalMs = nowMs
        val (low, high) = thermalView.range
        val max = frame.celsius.max()
        val text = String.format(
            Locale.US, "%.1f–%.1f °C   |   max %.1f °C   |   sensor %.1f °C   |   %.0f fps",
            low, high, max, frame.ambientC, fps,
        )
        runOnUiThread { if (thermalMode) status.text = text }
    }

    override fun onStart() {
        super.onStart()
        startThermal()
    }

    override fun onStop() {
        super.onStop()
        stopRecording()
        stopThermal()
    }

    override fun onDestroy() {
        super.onDestroy()
        analysisExecutor.execute { module?.destroy() }
        analysisExecutor.shutdown()
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            cameraProvider = providerFuture.get()
            if (!thermalMode) bindCamera()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCamera() {
        val provider = cameraProvider ?: return
        // Same 4:3 stream shape for preview and analysis, so analysis pixels map straight onto the preview.
        val ratio = ResolutionSelector.Builder()
            .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
            .build()
        val preview = Preview.Builder().setResolutionSelector(ratio).build()
            .also { it.surfaceProvider = previewView.surfaceProvider }
        val analysis = ImageAnalysis.Builder()
            .setResolutionSelector(ratio)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
            .also { it.setAnalyzer(analysisExecutor, ::analyze) }
        val recorder = Recorder.Builder()
            .setQualitySelector(
                QualitySelector.from(Quality.FHD, FallbackStrategy.higherQualityOrLowerThan(Quality.FHD))
            )
            .setTargetVideoEncodingBitRate(BITRATE)
            .build()
        val capture = VideoCapture.withOutput(recorder)
        provider.unbindAll()
        videoCapture = try {
            camera = provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis, capture)
            capture
        } catch (e: IllegalArgumentException) {
            // Some cameras cannot run preview + analysis + video at once: keep the demo, lose Record.
            Log.w(TAG, "video capture unavailable alongside analysis", e)
            provider.unbindAll()
            camera = provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
            null
        }
        readFocalLength()
    }

    private fun analyze(image: ImageProxy) {
        val net = module
        if (net == null) {
            image.close()
            return
        }
        // Upright frame, then the centred square (same box the overlay draws).
        val rotation = image.imageInfo.rotationDegrees
        val sensorTsNs = image.imageInfo.timestamp
        val analysisNs = SystemClock.elapsedRealtimeNanos()
        val bufferW = image.width
        val bufferH = image.height
        val sensorToBuffer = FloatArray(9).also { image.imageInfo.sensorToBufferTransformMatrix.getValues(it) }
        val raw = image.toBitmap()
        image.close()
        val frame = if (rotation == 0) raw else
            Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, Matrix().apply { postRotate(rotation.toFloat()) }, true)
        val side = (minOf(frame.width, frame.height) * BOX_FRACTION).toInt()
        val left = (frame.width - side) / 2
        val top = (frame.height - side) / 2
        val crop = Bitmap.createScaledBitmap(Bitmap.createBitmap(frame, left, top, side, side), SIZE, SIZE, true)

        // RGB -> CHW float, ImageNet-normalised (matches segkit.datasets.hands.normalize).
        crop.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)
        val plane = SIZE * SIZE
        for (i in 0 until plane) {
            val p = pixels[i]
            input[i] = (((p shr 16) and 0xFF) / 255f - MEAN[0]) / STD[0]
            input[plane + i] = (((p shr 8) and 0xFF) / 255f - MEAN[1]) / STD[1]
            input[2 * plane + i] = ((p and 0xFF) / 255f - MEAN[2]) / STD[2]
        }

        val t0 = SystemClock.elapsedRealtime()
        val outputs = net.forward(EValue.from(Tensor.fromBlob(input, longArrayOf(1, 3, SIZE.toLong(), SIZE.toLong()))))
        val inferMs = SystemClock.elapsedRealtime() - t0
        val mask = outputs[0].toTensor().dataAsFloatArray
        val present = outputs[1].toTensor().dataAsFloatArray[0]

        val now = SystemClock.elapsedRealtime()
        val fps = if (lastFrameMs > 0) 1000f / (now - lastFrameMs) else 0f
        lastFrameMs = now
        val cam = intrinsics(sensorToBuffer, frame.width, frame.height, bufferW)
        val modelHand = present > PRESENT_THRESHOLD
        val vetoed = modelHand && !thermalAgrees(mask, left, top, side, cam)
        val handVisible = modelHand && !vetoed
        outlineView.update(frame.width, frame.height, left, top, side, if (handVisible) edgePoints(mask) else null)
        logger.onFrame(now, crop, input, mask, present, inferMs, handVisible, frame.width, frame.height, rotation)
        sessions.onCameraFrame(sensorTsNs, analysisNs, frame.width, frame.height, rotation, left, top, side, crop,
            mask, present, handVisible, inferMs, sensorToBuffer, bufferW, bufferH)

        intrinsicsForCalibration = cam
        val state = calibState
        if (state == CalibState.COLLECTING) collectCalibration(sensorTsNs, mask, left, top, side, cam, handVisible, now)
        val showFused = calibration != null && fusedMode && !thermalMode && state == CalibState.OFF && thermalLive()
        if (showFused) renderFused(frame, mask, handVisible, left, top, side, cam)

        val logging = if (logger.enabled) "   |   ${logger.dumpCount} dumped" else ""
        runOnUiThread {
            updateRecordingLabel()
            if (showFused != fusedVisible) {
                fusedVisible = showFused
                fusedView.visibility = if (showFused) View.VISIBLE else View.GONE
                if (showFused) thermalInset.visibility = View.GONE
            }
            if (thermalMode || state == CalibState.SOLVING) return@runOnUiThread
            if (state == CalibState.COLLECTING) {
                status.text = (if (calibNeedsDepth) DEPTH_PROMPT else CALIBRATE_PROMPT) +
                    if (handVisible) "" else "\n(no hand seen)"
                status.setTextColor(if (handVisible) Color.GREEN else Color.WHITE)
                return@runOnUiThread
            }
            val fused = if (showFused) "$fusedStatus   |   " else ""
            status.text = String.format(
                Locale.US, "%s%s  %.2f   |   %d ms   |   %.0f fps%s",
                fused, if (handVisible) "HAND" else if (vetoed) "NOT WARM" else "NO HAND", present, inferMs, fps, logging,
            )
            status.setTextColor(if (handVisible) Color.GREEN else Color.WHITE)
        }
    }

    /**
     * Outline of the single largest blob in the mask (prob > 0.5), as (u, v) pairs in model pixels.
     * The model can mark stray skin-coloured patches as separate blobs; there is only one hand, so
     * everything but the largest connected region is dropped.
     */
    private fun edgePoints(mask: FloatArray): FloatArray {
        val labels = IntArray(SIZE * SIZE)
        val queue = IntArray(SIZE * SIZE)
        var best = 0
        var bestArea = 0
        var next = 0
        for (start in 0 until SIZE * SIZE) {
            if (mask[start] <= 0.5f || labels[start] != 0) continue
            next++
            var head = 0
            var tail = 0
            queue[tail++] = start
            labels[start] = next
            while (head < tail) {
                val i = queue[head++]
                val u = i % SIZE
                val v = i / SIZE
                for (j in intArrayOf(if (u > 0) i - 1 else -1, if (u < SIZE - 1) i + 1 else -1,
                                     if (v > 0) i - SIZE else -1, if (v < SIZE - 1) i + SIZE else -1)) {
                    if (j >= 0 && labels[j] == 0 && mask[j] > 0.5f) {
                        labels[j] = next
                        queue[tail++] = j
                    }
                }
            }
            if (tail > bestArea) {
                bestArea = tail
                best = next
            }
        }
        val out = ArrayList<Float>()
        if (best == 0) return out.toFloatArray()
        fun inside(u: Int, v: Int) = u in 0 until SIZE && v in 0 until SIZE && labels[v * SIZE + u] == best
        for (v in 0 until SIZE) for (u in 0 until SIZE) {
            if (inside(u, v) && (!inside(u - 1, v) || !inside(u + 1, v) || !inside(u, v - 1) || !inside(u, v + 1))) {
                out.add(u + 0.5f)
                out.add(v + 0.5f)
            }
        }
        return out.toFloatArray()
    }

    /** Draws the analysis box and the hand outline, mapped from frame pixels onto the FIT_CENTER preview. */
    class OutlineOverlay(context: Context) : View(context) {
        private val boxPaint = Paint().apply {
            color = Color.WHITE
            style = Paint.Style.STROKE
            strokeWidth = 3f
            alpha = 160
        }
        private val outlinePaint = Paint().apply {
            color = Color.GREEN
            strokeWidth = 6f
            strokeCap = Paint.Cap.ROUND
        }
        @Volatile private var state: State? = null

        private class State(val frameW: Int, val frameH: Int, val left: Int, val top: Int, val side: Int,
                            val points: FloatArray?)

        fun update(frameW: Int, frameH: Int, left: Int, top: Int, side: Int, points: FloatArray?) {
            state = State(frameW, frameH, left, top, side, points)
            postInvalidate()
        }

        override fun onDraw(canvas: Canvas) {
            val s = state ?: return
            val scale = minOf(width.toFloat() / s.frameW, height.toFloat() / s.frameH)
            val offX = (width - s.frameW * scale) / 2
            val offY = (height - s.frameH * scale) / 2
            val boxL = offX + s.left * scale
            val boxT = offY + s.top * scale
            val boxS = s.side * scale
            canvas.drawRect(boxL, boxT, boxL + boxS, boxT + boxS, boxPaint)
            val pts = s.points ?: return
            val k = boxS / SIZE
            val mapped = FloatArray(pts.size)
            for (i in pts.indices step 2) {
                mapped[i] = boxL + pts[i] * k
                mapped[i + 1] = boxT + pts[i + 1] * k
            }
            canvas.drawPoints(mapped, outlinePaint)
        }
    }

    companion object {
        private const val TAG = "HandDemo"
        private const val MODEL_ASSET = "handseg.pte"
        private const val SIZE = 384
        private const val BOX_FRACTION = 0.9f
        private const val PRESENT_THRESHOLD = 0.5f
        private const val BITRATE = 20_000_000  // high bitrate keeps finger edges free of compression mush
        private const val MENU_RECORD = 1
        private const val MENU_DUMP = 2
        private const val MENU_THERMAL = 3
        private const val MENU_CALIBRATE = 4
        private const val MENU_FUSED = 5
        private const val CALIBRATION_MS = 5000L
        private const val MAX_CALIBRATION_MS = 15000L
        private const val MIN_DEPTH_SPREAD = 1.35
        private const val DEPTH_PROMPT = "Almost there: move your hand closer to the phone, then further away"
        private const val VETO_GRID = 24
        private const val VETO_DELTA_C = 2.5f
        private const val CALIBRATE_PROMPT = "Hold your hand up inside the box to calibrate\n" +
            "Move it around slowly, nearer and further"
        private const val INSET_W = 360
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)

        /** ExecuTorch loads from a file path: copy the bundled model out of the APK (every launch, so a
         *  reinstalled APK never runs a stale model). */
        fun assetFilePath(context: Context, name: String): String {
            val file = File(context.filesDir, name)
            context.assets.open(name).use { input -> file.outputStream().use { input.copyTo(it) } }
            return file.absolutePath
        }
    }
}
