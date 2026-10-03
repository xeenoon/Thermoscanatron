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
import org.pytorch.executorch.Module
import org.json.JSONArray
import org.json.JSONObject
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
 *  - Calibrate thermal ↔ camera: the hand near, far, then top-left and bottom-left at medium distance, seen
 *    by both cameras -> [ThermalCalibration] finds where the thermal camera sits (angles, offsets), shown
 *    in a popup and saved ([CalibrationStore]);
 *  - Fused thermal view (once calibrated): only the part of the camera image the thermal camera also sees,
 *    with temperatures upsampled along the camera's edges ([FusionRenderer]); the rest is black.
 * Whenever the thermal camera is plugged in (and not fused), a small live thermal view sits in the corner.
 */
class DemoActivity : ComponentActivity() {
    private lateinit var previewView: PreviewView
    private lateinit var outlineView: OutlineOverlay
    private lateinit var status: TextView

    private val analysisExecutor = Executors.newSingleThreadExecutor()
    private var hands: HandTwoTier? = null
    private val input = FloatArray(3 * SIZE * SIZE)
    private val pixels = IntArray(SIZE * SIZE)
    private val pixelsSmall = IntArray(SMALL_SIZE * SMALL_SIZE)
    private val edgeLabels = IntArray(SMALL_SIZE * SMALL_SIZE)
    private val edgeQueue = IntArray(SMALL_SIZE * SMALL_SIZE)
    private val timing = LongArray(6)         // running per-stage ms (EMA x 16), see [analyze]
    private var timedFrames = 0L
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
    private var handDepthCm = Double.NaN      // analysis thread
    /** Phone camera focal length in sensor (active array) pixels, from Camera2; 0 until bound. */
    @Volatile private var sensorFocalPx = 0.0

    private enum class CalibState { OFF, COLLECTING, SOLVING }
    @Volatile private var calibState = CalibState.OFF
    private val calibCams = ArrayList<ThermalCalibration.CameraSample>()      // analysis thread only
    private val calibThermals = ConcurrentLinkedQueue<ThermalCalibration.ThermalSample>()
    @Volatile private var collectThermal = false
    /** Guided capture: near and far give the parallax that separates tilt from offset, the corners the roll. */
    private enum class CalibPhase(val prompt: String) {
        NEAR("Hold your open hand close to the sensor (~15–20 cm)"),
        FAR("Now hold it far away from the sensor (arm's length)"),
        TOP_LEFT("Medium distance (~30 cm): move your hand to the top-left of the box"),
        BOTTOM_LEFT("Medium distance (~30 cm): move your hand to the bottom-left of the box"),
    }
    @Volatile private var calibPhase = CalibPhase.NEAR
    @Volatile private var calibHint = ""
    private var calibPhaseMs = 0L
    private var calibStartMs = 0L
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
                hands = HandTwoTier(assetFilePath(this, SMALL_MODEL_ASSET), assetFilePath(this, MODEL_ASSET), SIZE,
                    SMALL_SIZE, prevMaskInput = false)
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
            calibPhase = CalibPhase.NEAR
            calibHint = ""
            calibPhaseMs = 0L
            calibStartMs = 0L
            calibLastHandMs = 0L
        }
        calibThermals.clear()
        collectThermal = true
        calibState = CalibState.COLLECTING
        calibProgress.isIndeterminate = false
        calibProgress.progress = 0
        calibProgress.visibility = View.VISIBLE
        status.setTextColor(Color.WHITE)
        status.text = calibPrompt()
    }

    private fun calibPrompt() = "Calibrating, step ${calibPhase.ordinal + 1}/${CalibPhase.values().size}\n${calibPhase.prompt}"

    /** Empty if [s] is where [calibPhase] wants the hand, else what to change. */
    private fun phaseHint(s: ThermalCalibration.CameraSample, cam: CameraIntrinsics): String {
        val z = s.depth
        val rx = (s.point[0] / z * cam.f + cam.cx - s.left) / s.side - 0.5
        val ry = (s.point[1] / z * cam.f + cam.cy - s.top) / s.side - 0.5
        fun depth(min: Double, max: Double) = when {
            z < min -> "further away"
            z > max -> "closer"
            else -> ""
        }
        fun corner(wantTop: Boolean): String {
            val d = depth(CALIB_MID_MIN_CM, CALIB_MID_MAX_CM)
            if (d.isNotEmpty()) return d
            val moves = listOfNotNull(
                if (rx > -CALIB_CORNER) "left" else null,
                if (wantTop && ry > -CALIB_CORNER) "up" else null,
                if (!wantTop && ry < CALIB_CORNER) "down" else null)
            return moves.joinToString(" and ")
        }
        return when (calibPhase) {
            CalibPhase.NEAR -> depth(0.0, CALIB_NEAR_CM)
            CalibPhase.FAR -> depth(CALIB_FAR_CM, Double.POSITIVE_INFINITY)
            CalibPhase.TOP_LEFT -> corner(true)
            CalibPhase.BOTTOM_LEFT -> corner(false)
        }
    }

    /**
     * Analysis thread: keep every frame where both cameras see the hand; each [CalibPhase] completes after
     * [CALIB_PHASE_MS] of the hand where that phase wants it.
     */
    private fun collectCalibration(tsNs: Long, mask: FloatArray, left: Int, top: Int, side: Int,
                                   cam: CameraIntrinsics, handVisible: Boolean, nowMs: Long) {
        if (calibStartMs == 0L) calibStartMs = nowMs
        if (nowMs - calibStartMs > CALIB_TIMEOUT_MS) {
            calibState = CalibState.SOLVING
            collectThermal = false
            finishCalibration(null, "Timed out at step ${calibPhase.ordinal + 1}: ${calibPhase.prompt.lowercase()}.")
            return
        }
        val sample = if (handVisible && thermalLive())
            ThermalCalibration.cameraSample(tsNs, mask, SIZE, left, top, side, cam) else null
        if (sample == null) {
            calibLastHandMs = 0L
            calibHint = ""
            return
        }
        calibCams.add(sample)
        val hint = phaseHint(sample, cam)
        calibHint = hint
        if (hint.isEmpty() && calibLastHandMs > 0) calibPhaseMs += minOf(nowMs - calibLastHandMs, 200L)
        calibLastHandMs = nowMs
        if (calibPhaseMs >= CALIB_PHASE_MS) {
            calibPhaseMs = 0L
            val next = calibPhase.ordinal + 1
            if (next < CalibPhase.values().size) calibPhase = CalibPhase.values()[next] else {
                calibState = CalibState.SOLVING
                val cams = ArrayList(calibCams)
                Thread({ solveCalibration(cams) }, "calibration").start()
                return
            }
        }
        val progress = ((calibPhase.ordinal + calibPhaseMs.toDouble() / CALIB_PHASE_MS) * 1000 / CalibPhase.values().size).toInt()
        runOnUiThread { calibProgress.progress = progress }
    }

    private fun solveCalibration(cams: List<ThermalCalibration.CameraSample>) {
        runOnUiThread { status.text = "Calibrating — solving for the thermal camera's position…" }
        // Thermal packets arrive a few hundred ms after the matching camera frame: wait for the last ones.
        SystemClock.sleep(600)
        collectThermal = false
        val thermals = calibThermals.toList()
        val camera = cams.firstOrNull()?.let { intrinsicsForCalibration } ?: return finishCalibration(null)
        val spread = ThermalCalibration.depthSpread(cams)
        if (spread < ThermalCalibration.MIN_DEPTH_SPREAD) return finishCalibration(null, String.format(Locale.US,
            "The hand was only %.1fx further away at its far end than at its near end (needs %.1fx): " +
                "go closer for step 1 and further for step 2.", spread, ThermalCalibration.MIN_DEPTH_SPREAD))
        try {
            val dir = File(getExternalFilesDir(null), "calibrations").apply { mkdirs() }
            val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            ThermalCalibration.writeCapture(File(dir, "capture_$stamp.bin"), cams, thermals, camera)
        } catch (e: Exception) {
            Log.w(TAG, "could not save the calibration capture", e)
        }
        val result = try {
            ThermalCalibration.solve(cams, thermals, camera, rig = ThermalCalibration.Rig.TAPED) { p ->
                runOnUiThread { calibProgress.progress = (p * 1000).toInt() }
            }
        } catch (e: Exception) {
            Log.e(TAG, "calibration failed", e)
            null
        }
        finishCalibration(result)
    }

    @Volatile private var intrinsicsForCalibration: CameraIntrinsics? = null

    private fun finishCalibration(result: ThermalCalibration.Result?, failure: String? = null) = runOnUiThread {
        calibState = CalibState.OFF
        calibProgress.visibility = View.GONE
        val dialog = AlertDialog.Builder(this)
        if (result == null) {
            dialog.setTitle("Not enough data")
                .setMessage(failure ?: ("Both cameras need to see your hand at every step. Keep it inside the box, " +
                    "in front of the thermal camera, and move it slowly."))
                .setPositiveButton("Retry") { _, _ -> startCalibration() }
                .setNegativeButton("Cancel", null)
        } else {
            val good = result.correlation >= 0.6
            dialog.setTitle(if (good) "Calibrated" else "Calibration looks poor")
                .setMessage(CalibrationStore.describe(result) + if (good) "" else
                    "\n\nTry again, holding your open hand flat towards the phone at each step.")
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
    private fun thermalAgrees(mask: FloatArray, n: Int, left: Int, top: Int, side: Int, cam: CameraIntrinsics): Boolean {
        if (!thermalLive()) return true
        val t = latestThermal?.celsius ?: return true
        val cal = calibration
        if (cal == null || calibState != CalibState.OFF) return ThermalCalibration.warmth(t) != null
        var above = 0
        for (p in mask) if (p > MASK_THRESHOLD) above++
        val areaPx = above.toDouble() / mask.size * side * side
        if (areaPx < 500) return true
        val z = cam.f * sqrt(ThermalCalibration.HAND_AREA_CM2 / areaPx)
        val uv = DoubleArray(2)
        val temps = ArrayList<Float>()
        for (gy in 0 until VETO_GRID) for (gx in 0 until VETO_GRID) {
            val mx = ((gx + 0.5) / VETO_GRID * n).toInt()
            val my = ((gy + 0.5) / VETO_GRID * n).toInt()
            if (mask[my * n + mx] <= MASK_THRESHOLD) continue
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
        var hand: FloatArray? = null
        if (handVisible) {
            var above = 0
            for (p in mask) if (p > MASK_THRESHOLD) above++
            val areaPx = above.toDouble() / mask.size * side * side
            if (areaPx > 500) {
                // The area-based depth jitters frame to frame; smooth it so the hand mapping does not wobble.
                val d = cam.f * sqrt(ThermalCalibration.HAND_AREA_CM2 / areaPx)
                handDepthCm = if (handDepthCm.isNaN()) d else handDepthCm + 0.3 * (d - handDepthCm)
                hand = mask.copyOf()
            }
        } else {
            handDepthCm = Double.NaN
        }
        val depth = if (hand != null) handDepthCm else null
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
        analysisExecutor.execute { hands?.close() }
        analysisExecutor.shutdown()
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            cameraProvider = providerFuture.get()
            if (!thermalMode) bindCamera()
        }, ContextCompat.getMainExecutor(this))
    }

    @OptIn(markerClass = [ExperimentalCamera2Interop::class])
    private fun fixedFrameRate(preview: Preview.Builder, analysis: ImageAnalysis.Builder) {
        val range = android.util.Range(TARGET_FPS, TARGET_FPS)
        val key = android.hardware.camera2.CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE
        androidx.camera.camera2.interop.Camera2Interop.Extender(preview).setCaptureRequestOption(key, range)
        androidx.camera.camera2.interop.Camera2Interop.Extender(analysis).setCaptureRequestOption(key, range)
    }

    private fun bindCamera() {
        val provider = cameraProvider ?: return
        // Same 4:3 stream shape for preview and analysis, so analysis pixels map straight onto the preview.
        val ratio = ResolutionSelector.Builder()
            .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
            .build()
        val previewB = Preview.Builder().setResolutionSelector(ratio)
        val analysisB = ImageAnalysis.Builder()
            .setResolutionSelector(ratio)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
        // The camera's top rate (30 fps on the A35) even in dim light: auto-exposure would otherwise drop to 15 fps,
        // and shorter exposures also blur a moving hand less.
        fixedFrameRate(previewB, analysisB)
        val preview = previewB.build().also { it.surfaceProvider = previewView.surfaceProvider }
        val analysis = analysisB.build().also { it.setAnalyzer(analysisExecutor, ::analyze) }
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

    /** The upright frame's centred box, rotated and scaled straight out of the camera buffer in one draw. */
    private fun renderCrop(raw: Bitmap, rotation: Int, left: Int, top: Int, side: Int, out: Int): Bitmap {
        val m = Matrix()
        m.postRotate(rotation.toFloat())
        when (rotation) {
            90 -> m.postTranslate(raw.height.toFloat(), 0f)
            180 -> m.postTranslate(raw.width.toFloat(), raw.height.toFloat())
            270 -> m.postTranslate(0f, raw.width.toFloat())
        }
        m.postTranslate(-left.toFloat(), -top.toFloat())
        m.postScale(out.toFloat() / side, out.toFloat() / side)
        val bmp = Bitmap.createBitmap(out, out, Bitmap.Config.ARGB_8888)
        Canvas(bmp).drawBitmap(raw, m, Paint(Paint.FILTER_BITMAP_FLAG))
        return bmp
    }

    /** Small-model mask (SMALL_SIZE^2) -> SIZE^2 for the consumers that work in model pixels, bilinear. */
    private fun upsample(m: FloatArray): FloatArray {
        val n = SMALL_SIZE
        val out = FloatArray(SIZE * SIZE)
        val r = n.toFloat() / SIZE
        for (y in 0 until SIZE) {
            val sy = ((y + 0.5f) * r - 0.5f).coerceIn(0f, n - 1f)
            val y0 = sy.toInt(); val y1 = minOf(y0 + 1, n - 1); val fy = sy - y0
            for (x in 0 until SIZE) {
                val sx = ((x + 0.5f) * r - 0.5f).coerceIn(0f, n - 1f)
                val x0 = sx.toInt(); val x1 = minOf(x0 + 1, n - 1); val fx = sx - x0
                val t = m[y0 * n + x0] * (1 - fx) + m[y0 * n + x1] * fx
                val b = m[y1 * n + x0] * (1 - fx) + m[y1 * n + x1] * fx
                out[y * SIZE + x] = t * (1 - fy) + b * fy
            }
        }
        return out
    }

    private fun normaliseBig(): FloatArray {
        val plane = SIZE * SIZE
        for (i in 0 until plane) {
            val p = pixels[i]
            input[i] = (((p shr 16) and 0xFF) / 255f - MEAN[0]) / STD[0]
            input[plane + i] = (((p shr 8) and 0xFF) / 255f - MEAN[1]) / STD[1]
            input[2 * plane + i] = ((p and 0xFF) / 255f - MEAN[2]) / STD[2]
        }
        return input
    }

    /**
     * Per frame: the small model's crop is drawn straight from the camera buffer; the full upright frame, the
     * 384 px crop and the 384 px mask are only made when something needs them (big model free, recording,
     * logging, calibration, fused view). Stage times are logged every frame and averaged in the status line.
     */
    private fun analyze(image: ImageProxy) {
        val net = hands
        if (net == null) {
            image.close()
            return
        }
        val tStart = SystemClock.elapsedRealtimeNanos()
        val rotation = image.imageInfo.rotationDegrees
        val sensorTsNs = image.imageInfo.timestamp
        val analysisNs = SystemClock.elapsedRealtimeNanos()
        val bufferW = image.width
        val bufferH = image.height
        val sensorToBuffer = FloatArray(9).also { image.imageInfo.sensorToBufferTransformMatrix.getValues(it) }
        val raw = image.toBitmap()
        image.close()
        val fw = if (rotation % 180 == 0) raw.width else raw.height
        val fh = if (rotation % 180 == 0) raw.height else raw.width
        val side = (minOf(fw, fh) * BOX_FRACTION).toInt()
        val left = (fw - side) / 2
        val top = (fh - side) / 2
        val frame by lazy {
            if (rotation == 0) raw else
                Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, Matrix().apply { postRotate(rotation.toFloat()) }, true)
        }
        val crop by lazy {
            renderCrop(raw, rotation, left, top, side, SIZE).also { it.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE) }
        }
        renderCrop(raw, rotation, left, top, side, SMALL_SIZE).getPixels(pixelsSmall, 0, SMALL_SIZE, 0, 0, SMALL_SIZE, SMALL_SIZE)
        val tPrep = SystemClock.elapsedRealtimeNanos()

        // Small model every frame; the big model checks its hand/no-hand call in the background ([HandTwoTier]).
        val result = net.run(pixelsSmall, { crop; pixels }, MEAN, STD)
        val inferMs = result.smallMs
        val maskSmall = result.mask
        val mask by lazy { upsample(maskSmall) }
        val present = result.present
        val tModel = SystemClock.elapsedRealtimeNanos()

        val now = SystemClock.elapsedRealtime()
        val fps = if (lastFrameMs > 0) 1000f / (now - lastFrameMs) else 0f
        lastFrameMs = now
        val cam = intrinsics(sensorToBuffer, fw, fh, bufferW)
        val modelHand = present > PRESENT_THRESHOLD
        val vetoed = modelHand && !thermalAgrees(maskSmall, SMALL_SIZE, left, top, side, cam)
        val handVisible = modelHand && !vetoed
        outlineView.update(fw, fh, left, top, side, if (handVisible) edgePoints(maskSmall) else null)
        val tOutline = SystemClock.elapsedRealtimeNanos()
        if (logger.enabled) {
            logger.onFrame(now, crop, normaliseBig(), mask, present, inferMs, handVisible, fw, fh, rotation)
        }
        if (sessions.active) {
            sessions.onCameraFrame(sensorTsNs, analysisNs, fw, fh, rotation, left, top, side, crop,
                mask, present, handVisible, inferMs, sensorToBuffer, bufferW, bufferH)
        }
        val tRecord = SystemClock.elapsedRealtimeNanos()

        intrinsicsForCalibration = cam
        val state = calibState
        if (state == CalibState.COLLECTING) collectCalibration(sensorTsNs, mask, left, top, side, cam, handVisible, now)
        val showFused = calibration != null && fusedMode && !thermalMode && state == CalibState.OFF && thermalLive()
        if (showFused) renderFused(frame, mask, handVisible, left, top, side, cam)
        val tEnd = SystemClock.elapsedRealtimeNanos()
        val stages = longArrayOf(tPrep - tStart, tModel - tPrep, tOutline - tModel, tRecord - tOutline,
            tEnd - tRecord, tEnd - tStart)
        for (i in stages.indices) {
            val ms = stages[i] / 1_000_000
            timing[i] = if (timedFrames == 0L) ms * 16 else timing[i] - timing[i] / 16 + ms
        }
        timedFrames++
        Log.i(TAG, String.format(Locale.US,
            "timing prep=%d model=%d (small %d, big %d) outline=%d record=%d rest=%d total=%d ms  p=%.2f small=%.2f big=%s",
            stages[0] / 1_000_000, stages[1] / 1_000_000, inferMs, result.bigMs, stages[2] / 1_000_000,
            stages[3] / 1_000_000, stages[4] / 1_000_000, stages[5] / 1_000_000, present, result.smallPresent,
            result.bigPresent?.let { String.format(Locale.US, "%.2f", it) } ?: "-"))

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
                val hint = calibHint
                status.text = calibPrompt() + when {
                    !handVisible -> "\n(no hand seen)"
                    hint.isNotEmpty() -> "\n(move $hint)"
                    else -> ""
                }
                status.setTextColor(if (handVisible && hint.isEmpty()) Color.GREEN else Color.WHITE)
                return@runOnUiThread
            }
            val fused = if (showFused) "$fusedStatus   |   " else ""
            status.text = String.format(
                Locale.US, "%s%s  %.2f   |   frame %d ms (prep %d, small %d, big %d bg)   |   %.0f fps%s",
                fused, if (handVisible) "SKIN" else if (vetoed) "NOT WARM" else "NO SKIN", present,
                timing[5] / 16, timing[0] / 16, inferMs, result.bigMs, fps, logging,
            )
            status.setTextColor(if (handVisible) Color.GREEN else Color.WHITE)
        }
    }

    /**
     * Outline of every skin region in the small model's mask (prob > 0.5), as (u, v) pairs in SIZE-model pixels
     * (the overlay's units): a face, two hands and an arm are separate regions. Specks smaller than
     * [MIN_REGION] of the crop are dropped. No per-pixel allocation: this runs on every frame.
     */
    private fun edgePoints(mask: FloatArray): FloatArray {
        val n = SMALL_SIZE
        val labels = edgeLabels
        val queue = edgeQueue
        labels.fill(0)
        val keep = BooleanArray(n * n / 4 + 2)   // per region label: big enough to draw
        var next = 0
        for (start in 0 until n * n) {
            if (mask[start] <= MASK_THRESHOLD || labels[start] != 0) continue
            next++
            if (next >= keep.size) break
            var head = 0
            var tail = 0
            queue[tail++] = start
            labels[start] = next
            while (head < tail) {
                val i = queue[head++]
                val u = i % n
                val v = i / n
                if (u > 0 && labels[i - 1] == 0 && mask[i - 1] > MASK_THRESHOLD) { labels[i - 1] = next; queue[tail++] = i - 1 }
                if (u < n - 1 && labels[i + 1] == 0 && mask[i + 1] > MASK_THRESHOLD) { labels[i + 1] = next; queue[tail++] = i + 1 }
                if (v > 0 && labels[i - n] == 0 && mask[i - n] > MASK_THRESHOLD) { labels[i - n] = next; queue[tail++] = i - n }
                if (v < n - 1 && labels[i + n] == 0 && mask[i + n] > MASK_THRESHOLD) { labels[i + n] = next; queue[tail++] = i + n }
            }
            keep[next] = tail >= MIN_REGION * n * n
        }
        if (next == 0) return FloatArray(0)
        fun inside(u: Int, v: Int) = u in 0 until n && v in 0 until n && labels[v * n + u].let { it != 0 && keep[it] }
        var count = 0
        for (v in 0 until n) for (u in 0 until n) {
            if (inside(u, v) && (!inside(u - 1, v) || !inside(u + 1, v) || !inside(u, v - 1) || !inside(u, v + 1))) count++
        }
        val out = FloatArray(2 * count)
        val k = SIZE.toFloat() / n
        var j = 0
        for (v in 0 until n) for (u in 0 until n) {
            if (inside(u, v) && (!inside(u - 1, v) || !inside(u + 1, v) || !inside(u, v - 1) || !inside(u, v + 1))) {
                out[j++] = (u + 0.5f) * k
                out[j++] = (v + 0.5f) * k
            }
        }
        return out
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
        private const val MODEL_ASSET = "skinseg.pte"            // skin: faces, hands, arms (segkit-label-skin)
        private const val SMALL_MODEL_ASSET = "skinseg_small.pte"
        private const val SMALL_SIZE = 256
        // Skin probability needed to draw a pixel: 0.8 keeps the gaps between fingers open and drops faint desk/cable
        // blobs (desk recording: finger gaps filled 24% -> 12%, false skin 1.1% -> 0.8% of the frame).
        private const val MASK_THRESHOLD = 0.8f
        private const val MIN_REGION = 0.003   // of the crop: smaller skin specks are not drawn
        private const val TARGET_FPS = 30
        private const val SIZE = 384
        private const val BOX_FRACTION = 0.9f
        private const val PRESENT_THRESHOLD = 0.5f
        private const val BITRATE = 20_000_000  // high bitrate keeps finger edges free of compression mush
        private const val MENU_RECORD = 1
        private const val MENU_DUMP = 2
        private const val MENU_THERMAL = 3
        private const val MENU_CALIBRATE = 4
        private const val MENU_FUSED = 5
        private const val CALIB_PHASE_MS = 3000L        // hand time in the right place per step
        private const val CALIB_TIMEOUT_MS = 90_000L
        private const val CALIB_NEAR_CM = 20.0
        private const val CALIB_FAR_CM = 40.0
        private const val CALIB_MID_MIN_CM = 22.0
        private const val CALIB_MID_MAX_CM = 40.0
        private const val CALIB_CORNER = 0.12           // hand centre this far (box fraction) off-centre
        private const val VETO_GRID = 24
        private const val VETO_DELTA_C = 2.5f
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
