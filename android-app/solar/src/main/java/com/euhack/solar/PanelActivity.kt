package com.euhack.solar

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Path
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.hardware.camera2.CameraCharacteristics
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
import androidx.annotation.OptIn
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.euhack.hello.CalibrationStore
import com.euhack.hello.CameraIntrinsics
import com.euhack.hello.ThermalCalibration
import com.euhack.hello.ThermalFrame
import com.euhack.hello.ThermalUsbStream
import com.euhack.hello.BlockMotion
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.File
import java.util.Locale
import java.util.concurrent.Executors
import kotlin.math.abs

/**
 * Solar panel cells, live: PanelNet (ExecuTorch) on the centred square of each camera frame, [PanelTracker] for
 * which cell is which (gyro rotation as the motion prior), and with the USB thermal camera plugged in,
 * [PanelThermal] for each cell's temperature and hotspots (|cell - panel average| > 5 C).
 *
 * Step back until the whole panel (or at least a corner of it) is in the box once: it locks on, then keeps
 * the cell numbers while you move in close. The cyan outline is what the thermal camera sees; sweep it over
 * the panel to fill in the cells.
 *
 * Thermal frames arrive [ThermalCalibration.Result.latencyMs] after the scene they show, so each one is paired
 * with the panel geometry of the camera frame captured that long before it arrived.
 * The camera <-> thermal calibration is the hand app's ([CalibrationStore] format): <files>/thermal_calib.json,
 * seeded from the bundled asset (the phone's own calibration when this app was built).
 */
class PanelActivity : ComponentActivity(), SensorEventListener {
    private lateinit var previewView: PreviewView
    private lateinit var cellOverlay: CellOverlay
    private lateinit var status: TextView
    private lateinit var detail: TextView

    private val analysisExecutor = Executors.newSingleThreadExecutor()
    // Fast path (every camera frame): the small model. Slow path (own thread, whenever it is free): the big one.
    private var small: Module? = null
    private var big: Module? = null
    private val inputSmall = FloatArray(3 * SMALL * SMALL)
    private val pixelsSmall = IntArray(SMALL * SMALL)
    private val inputBig = FloatArray(3 * SIZE * SIZE)
    private val pixelsBig = IntArray(SIZE * SIZE)
    private val bigExecutor = Executors.newSingleThreadExecutor { r ->
        Thread({ android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_BACKGROUND); r.run() }, "panel-big")
    }
    private val bigBusy = java.util.concurrent.atomic.AtomicBoolean(false)
    private val bigResults = java.util.concurrent.ConcurrentLinkedQueue<Pair<Long, DoubleArray?>>()
    private val slowTracker by lazy { PanelTracker(spec, SIZE) }
    @Volatile private var bigMs = 0L
    private val fastHistory = LinkedHashMap<Long, DoubleArray?>()
    private var frameIndex = 0L
    private val gyro = GyroTrack()
    private var prevFrameNs = 0L
    /** Which gyro prior matches the image (+1: R^T, -1: R, 0: not known yet); learned from the fits. */
    @Volatile private var gyroSign = 0
    private var gyroVote = 0.0
    private val spec = PanelSpec()
    private val tracker = PanelTracker(spec, SIZE)
    private val motion = BlockMotion(SMALL, 2)
    private var thermal: PanelThermal? = null
    private var calibration: ThermalCalibration.Result? = null
    private var stream: ThermalUsbStream? = null
    @Volatile private var latestThermal: ThermalFrame? = null
    private var processedThermal: ThermalFrame? = null
    @Volatile private var thermalStatus = "thermal: not connected"
    @Volatile private var resetRequested = false
    private var sensorFocalPx = 0.0
    private var recorder: DebugRecorder? = null
    private var lastFrameMs = 0L

    /** Recent camera frames' panel geometry, for pairing thermal frames with the scene they show. */
    private class Pose(val sensorNs: Long, val hFrame: DoubleArray?)
    private val history = ArrayDeque<Pose>()


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
        cellOverlay = CellOverlay(this, spec, gyro)
        status = TextView(this).apply {
            textSize = 17f
            setTextColor(Color.WHITE)
            setBackgroundColor(0x99000000.toInt())
            setPadding(24, 16, 24, 16)
            gravity = Gravity.CENTER
            text = "Loading model…"
        }
        detail = TextView(this).apply {
            textSize = 13f
            setTextColor(Color.WHITE)
            setBackgroundColor(0x99000000.toInt())
            setPadding(24, 12, 24, 12)
            gravity = Gravity.CENTER
        }
        val rec = Button(this).apply {
            text = "Rec"
            setOnClickListener {
                analysisExecutor.execute {
                    val r = recorder
                    if (r == null) {
                        recorder = DebugRecorder(File(getExternalFilesDir(null), "panel_debug"))
                        runOnUiThread { text = "Stop" }
                    } else {
                        r.close()
                        recorder = null
                        runOnUiThread { text = "Rec" }
                    }
                }
            }
        }
        val reset = Button(this).apply {
            text = "Reset"
            setOnClickListener {
                resetRequested = true
            }
        }
        setContentView(FrameLayout(this).apply {
            setBackgroundColor(Color.BLACK)
            addView(previewView, FrameLayout.LayoutParams(-1, -1))
            addView(cellOverlay, FrameLayout.LayoutParams(-1, -1))
            addView(LinearLayout(context).apply {
                orientation = LinearLayout.VERTICAL
                addView(status, LinearLayout.LayoutParams(-1, -2))
                addView(detail, LinearLayout.LayoutParams(-1, -2))
            }, FrameLayout.LayoutParams(-1, -2, Gravity.TOP).apply { topMargin = 110 })
            addView(LinearLayout(context).apply {
                addView(rec)
                addView(reset)
            }, FrameLayout.LayoutParams(-2, -2, Gravity.BOTTOM or Gravity.CENTER_HORIZONTAL)
                .apply { bottomMargin = 180 })
        })

        loadCalibration()
        analysisExecutor.execute {
            try {
                // Separate thread budgets so the background model cannot starve the one the camera waits for.
                small = Module.load(assetFilePath(this, SMALL_ASSET), Module.LOAD_MODE_FILE, SMALL_THREADS)
                big = Module.load(assetFilePath(this, MODEL_ASSET), Module.LOAD_MODE_FILE, BIG_THREADS)
                runOnUiThread { status.text = "Point at the panel: whole panel (or a corner) in the box" }
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

    private fun loadCalibration() {
        val file = File(filesDir, "thermal_calib.json")
        if (!file.isFile) assets.open(CALIB_ASSET).use { input -> file.outputStream().use { input.copyTo(it) } }
        calibration = CalibrationStore(filesDir).load()
        thermal = calibration?.let { PanelThermal(spec, it.pose) }
        if (calibration == null) thermalStatus = "thermal: no calibration"
    }

    override fun onStart() {
        super.onStart()
        stream = ThermalUsbStream(this, { latestThermal = it }) { text -> thermalStatus = "thermal: $text" }
            .also { it.start() }
        val sm = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        sm.getDefaultSensor(Sensor.TYPE_GYROSCOPE)?.let { sm.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
    }

    override fun onStop() {
        super.onStop()
        stream?.stop()
        stream = null
        (getSystemService(Context.SENSOR_SERVICE) as SensorManager).unregisterListener(this)
    }

    override fun onDestroy() {
        super.onDestroy()
        analysisExecutor.execute { small?.destroy() }
        analysisExecutor.shutdown()
        bigExecutor.execute { big?.destroy() }
        bigExecutor.shutdown()
    }

    override fun onSensorChanged(event: SensorEvent) = gyro.onEvent(event)

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val ratio = ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                .build()
            val previewB = Preview.Builder().setResolutionSelector(ratio)
            val analysisB = ImageAnalysis.Builder()
                .setResolutionSelector(ratio)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            fullFrameRate(previewB, analysisB)
            val preview = previewB.build().also { it.surfaceProvider = previewView.surfaceProvider }
            val analysis = analysisB.build().also { it.setAnalyzer(analysisExecutor, ::analyze) }
            provider.unbindAll()
            val camera = provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
            readFocalLength(camera.cameraInfo)
        }, ContextCompat.getMainExecutor(this))
    }

    /** Ask for the camera's top frame rate (30 fps on the A35: 15-30 is all it offers) instead of a dim-light
     *  slowdown. Short exposures also mean less motion blur. */
    @OptIn(markerClass = [ExperimentalCamera2Interop::class])
    private fun fullFrameRate(vararg builders: Any) {
        val range = android.util.Range(TARGET_FPS, TARGET_FPS)
        for (b in builders) when (b) {
            is Preview.Builder -> androidx.camera.camera2.interop.Camera2Interop.Extender(b)
                .setCaptureRequestOption(android.hardware.camera2.CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, range)
            is ImageAnalysis.Builder -> androidx.camera.camera2.interop.Camera2Interop.Extender(b)
                .setCaptureRequestOption(android.hardware.camera2.CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, range)
        }
    }

    @OptIn(markerClass = [ExperimentalCamera2Interop::class])
    private fun readFocalLength(info: androidx.camera.core.CameraInfo) {
        val c2 = Camera2CameraInfo.from(info)
        val focal = c2.getCameraCharacteristic(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.firstOrNull()
        val size = c2.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
        val px = c2.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE)
        if (focal != null && size != null && px != null) sensorFocalPx = focal.toDouble() / size.width * px.width
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

    private fun normalise(px: IntArray, dst: FloatArray, n: Int) {
        val plane = n * n
        for (i in 0 until plane) {
            val p = px[i]
            dst[i] = (((p shr 16) and 0xFF) / 255f - MEAN[0]) / STD[0]
            dst[plane + i] = (((p shr 8) and 0xFF) / 255f - MEAN[1]) / STD[1]
            dst[2 * plane + i] = ((p and 0xFF) / 255f - MEAN[2]) / STD[2]
        }
    }

    private fun analyze(image: ImageProxy) {
        val fast = small
        if (fast == null || big == null) {
            image.close()
            return
        }
        if (resetRequested) {
            resetRequested = false
            tracker.reset()
            motion.reset()
            thermal?.reset()
            history.clear()
            fastHistory.clear()
            bigResults.clear()
        }
        val rotation = image.imageInfo.rotationDegrees
        val sensorNs = image.imageInfo.timestamp
        val bufferW = image.width
        val sensorToBuffer = FloatArray(9).also { image.imageInfo.sensorToBufferTransformMatrix.getValues(it) }
        val raw = image.toBitmap()
        image.close()
        val fw = if (rotation % 180 == 0) raw.width else raw.height
        val fh = if (rotation % 180 == 0) raw.height else raw.width
        val side = (minOf(fw, fh) * BOX_FRACTION).toInt()
        val left = (fw - side) / 2
        val top = (fh - side) / 2
        val frameNo = frameIndex++

        // Fast path: small model on a SMALL x SMALL crop.
        val cropSmall = renderCrop(raw, rotation, left, top, side, SMALL)
        cropSmall.getPixels(pixelsSmall, 0, SMALL, 0, 0, SMALL, SMALL)
        normalise(pixelsSmall, inputSmall, SMALL)
        val t0 = SystemClock.elapsedRealtime()
        val out = fast.forward(EValue.from(Tensor.fromBlob(inputSmall, longArrayOf(1, 3, SMALL.toLong(), SMALL.toLong()))))
        val inferMs = SystemClock.elapsedRealtime() - t0
        val dense = out[0].toTensor().dataAsFloatArray
        val present = out[1].toTensor().dataAsFloatArray[0]

        // Camera intrinsics in the upright frame; tracker coordinates are a SIZE x SIZE view of the box.
        val f = if (sensorFocalPx > 0) sensorFocalPx * sensorToBuffer[0] else bufferW / 2 / Math.tan(Math.toRadians(32.5))
        val cam = CameraIntrinsics(f, fw / 2.0, fh / 2.0)
        val s = SIZE.toDouble() / side
        val a = doubleArrayOf(s, 0.0, -left * s, 0.0, s, -top * s, 0.0, 0.0, 1.0)
        val aInv = Mat3.inv(a)
        val k = doubleArrayOf(cam.f, 0.0, cam.cx, 0.0, cam.f, cam.cy, 0.0, 0.0, 1.0)
        val kInv = Mat3.inv(k)

        // Motion priors: measured image shift, none, and the gyro rotation since the last frame (both signs until
        // the fits have shown which one is right).
        val w = if (prevFrameNs != 0L) gyro.between(gyro.at(prevFrameNs), gyro.at(sensorNs)) else DoubleArray(3)
        val blurred = prevFrameNs != 0L && gyro.speed(prevFrameNs, sensorNs) > BLUR_RAD_S
        prevFrameNs = sensorNs
        val shift = motion.update(pixelsSmall)?.let { Mat3.translate(it[0] * SIZE / SMALL, it[1] * SIZE / SMALL) }
        val r = Mat3.rotation(w[0], w[1], w[2])
        val gyroT = Mat3.mul(a, Mat3.mul(k, Mat3.mul(Mat3.transpose(r), Mat3.mul(kInv, aInv))))
        val gyroR = Mat3.mul(a, Mat3.mul(k, Mat3.mul(r, Mat3.mul(kInv, aInv))))
        val predictions = ArrayList<DoubleArray>()
        // First entry = what the tracker coasts on: the gyro once its sign is known (it sees through blur), else
        // the image shift.
        when (gyroSign) {
            1 -> predictions.add(gyroT)
            -1 -> predictions.add(gyroR)
        }
        shift?.let { predictions.add(it) }
        predictions.add(Mat3.identity())
        if (gyroSign == 0) { predictions.add(gyroT); predictions.add(gyroR) }

        val t1 = SystemClock.elapsedRealtime()
        val state = tracker.step(dense, present, predictions, denseSize = SMALL, coastOnly = blurred && gyroSign != 0)
        if (gyroSign == 0 && state == PanelTracker.State.TRACKING) {
            val fits = tracker.lastFits
            val n = fits.size
            if (n >= 2 && fits[n - 2] >= 0 && fits[n - 1] >= 0) {
                gyroVote += (fits[n - 2] - fits[n - 1])
                if (Math.abs(gyroVote) > 1.0) gyroSign = if (gyroVote > 0) 1 else -1
            }
        }

        // Slow path: hand this frame to the big model if it is free; apply any answers that have come back.
        if (bigBusy.compareAndSet(false, true)) {
            renderCrop(raw, rotation, left, top, side, SIZE).getPixels(pixelsBig, 0, SIZE, 0, 0, SIZE, SIZE)
            val seed = tracker.h?.copyOf()
            bigExecutor.execute { runBig(frameNo, seed) }
        }
        fastHistory[frameNo] = tracker.h?.copyOf()
        while (true) {
            val (t, hb) = bigResults.poll() ?: break
            if (hb == null) continue
            val ht = fastHistory[t]
            val now = tracker.h
            // Carry the big model's fix for frame t to now through the fast path's own motion since then.
            tracker.seed(if (ht != null && now != null) Mat3.mul(now, Mat3.mul(Mat3.inv(ht), hb)) else hb)
        }
        while (fastHistory.size > 90) fastHistory.remove(fastHistory.keys.first())
        val trackMs = SystemClock.elapsedRealtime() - t1
        val fits = tracker.lastFits.joinToString("/") { String.format(Locale.US, "%.2f", it) }
        Log.i(TAG, String.format(Locale.US,
            "frame state=%s present=%.2f samples=%d fits=%s blur=%b gyroSign=%d small=%dms big=%dms track=%dms cell=%s",
            state, present, tracker.lastSamples, fits, blurred, gyroSign, inferMs, bigMs, trackMs,
            tracker.cellAt(SIZE / 2.0, SIZE / 2.0)?.joinToString(",") ?: "-"))
        recorder?.write(cropSmall, sensorNs, w, state, present, tracker.lastSamples, fits, inferMs, trackMs)
        val hCrop = tracker.h
        val hFrame = hCrop?.let { Mat3.mul(aInv, it) }
        history.addLast(Pose(sensorNs, if (state == PanelTracker.State.COASTING) null else hFrame))
        while (history.size > 40) history.removeFirst()

        // Thermal: pair the newest thermal frame with the camera frame it shows.
        val th = thermal
        val cal = calibration
        var outline: FloatArray? = null
        val tf = latestThermal
        if (th != null && cal != null && tf != null && tf !== processedThermal) {
            processedThermal = tf
            val want = tf.receivedNs - cal.latencyMs * 1_000_000L
            val best = history.minByOrNull { abs(it.sensorNs - want) }
            if (best?.hFrame != null && abs(best.sensorNs - want) < PAIR_TOLERANCE_NS) {
                th.plane(best.hFrame, cam)?.let { th.update(tf.celsius, it) }
            }
        }
        if (th != null && hFrame != null) th.plane(hFrame, cam)?.let { outline = th.viewOutline(it, cam) }

        val centre = tracker.cellAt(SIZE / 2.0, SIZE / 2.0)
        cellOverlay.update(fw, fh, left, top, side, hFrame, state, centre, th, outline, sensorNs, k, gyroSign)

        val now = SystemClock.elapsedRealtime()
        val fps = if (lastFrameMs > 0) 1000f / (now - lastFrameMs) else 0f
        lastFrameMs = now
        val head = when (state) {
            PanelTracker.State.LOST -> if (present > 0.5f) "PANEL — step back to lock on" else "No panel"
            else -> centre?.let { c ->
                val k = c[0] * spec.cols + c[1]
                val t = th?.cellC?.get(k)?.takeIf { !it.isNaN() }
                "Cell (${c[0]},${c[1]})" + (t?.let { String.format(Locale.US, "  %.1f °C", it) } ?: "")
            } ?: "Panel locked — aim the cross at a cell"
        }
        val hot = th?.let { (0 until spec.rows * spec.cols).filter(it::isHotspot) } ?: emptyList()
        val panelLine = th?.panelC?.takeIf { !it.isNaN() }?.let {
            val seen = th.cellFrames.count { n -> n > 0 }
            String.format(Locale.US, "panel avg %.1f °C   cells seen %d/%d   hotspots %s",
                it, seen, spec.rows * spec.cols,
                if (hot.isEmpty()) "none" else hot.joinToString(" ") { k -> "(${k / spec.cols},${k % spec.cols})" })
        } ?: thermalStatus
        val info = String.format(Locale.US, "%s   |   %s   |   small %d ms, big %d ms   %.0f fps",
            panelLine, state.name.lowercase(), inferMs, bigMs, fps)
        runOnUiThread {
            status.text = head
            status.setTextColor(if (hot.isNotEmpty()) Color.rgb(255, 120, 120) else Color.WHITE)
            detail.text = info
        }
    }

    /** Big model on [pixelsBig] (frame [frameNo]); its tracker starts from what the fast path believed then. */
    private fun runBig(frameNo: Long, seed: DoubleArray?) {
        try {
            val net = big ?: return
            normalise(pixelsBig, inputBig, SIZE)
            val t0 = SystemClock.elapsedRealtime()
            val out = net.forward(EValue.from(Tensor.fromBlob(inputBig, longArrayOf(1, 3, SIZE.toLong(), SIZE.toLong()))))
            val dense = out[0].toTensor().dataAsFloatArray
            val present = out[1].toTensor().dataAsFloatArray[0]
            slowTracker.seed(seed)
            val st = slowTracker.step(dense, present, listOf(Mat3.identity()))
            bigMs = SystemClock.elapsedRealtime() - t0
            val good = st == PanelTracker.State.TRACKING || st == PanelTracker.State.RELOCKED ||
                st == PanelTracker.State.ACQUIRED
            bigResults.add(frameNo to if (good) slowTracker.h?.copyOf() else null)
        } catch (e: Exception) {
            Log.e(TAG, "big model failed", e)
        } finally {
            bigBusy.set(false)
        }
    }

    /** Grid, cell labels and temperatures, hotspots, thermal view outline, crosshair; FIT_CENTER like the preview. */
    class CellOverlay(context: Context, private val spec: PanelSpec, private val gyro: GyroTrack) : View(context) {
        private class St(val fw: Int, val fh: Int, val left: Int, val top: Int, val side: Int, val h: DoubleArray?,
                         val state: PanelTracker.State, val centre: IntArray?, val temps: DoubleArray?,
                         val deltas: DoubleArray?, val hot: BooleanArray?, val outline: FloatArray?,
                         val frameNs: Long, val k: DoubleArray, val gyroSign: Int)
        @Volatile private var st: St? = null
        private val grid = Paint().apply { color = Color.RED; strokeWidth = 4f; style = Paint.Style.STROKE; isAntiAlias = true }
        private val box = Paint().apply { color = Color.WHITE; alpha = 140; strokeWidth = 3f; style = Paint.Style.STROKE }
        private val label = Paint().apply { color = Color.YELLOW; isAntiAlias = true; isFakeBoldText = true
            setShadowLayer(4f, 0f, 0f, Color.BLACK) }
        private val temp = Paint().apply { color = Color.CYAN; isAntiAlias = true; isFakeBoldText = true
            setShadowLayer(4f, 0f, 0f, Color.BLACK) }
        private val hotFill = Paint().apply { color = Color.argb(110, 255, 0, 0); style = Paint.Style.FILL }
        private val coldFill = Paint().apply { color = Color.argb(110, 0, 90, 255); style = Paint.Style.FILL }
        private val centreFill = Paint().apply { color = Color.argb(70, 255, 255, 255); style = Paint.Style.FILL }
        private val view = Paint().apply { color = Color.CYAN; strokeWidth = 4f; style = Paint.Style.STROKE
            pathEffect = android.graphics.DashPathEffect(floatArrayOf(18f, 12f), 0f); isAntiAlias = true }
        private val cross = Paint().apply { color = Color.WHITE; strokeWidth = 4f }
        private val tmp = DoubleArray(2)

        private fun warp(m: DoubleArray, pts: FloatArray): FloatArray {
            val o = FloatArray(pts.size)
            for (i in 0 until pts.size / 2) {
                Mat3.apply(m, pts[2 * i].toDouble(), pts[2 * i + 1].toDouble(), tmp)
                o[2 * i] = tmp[0].toFloat(); o[2 * i + 1] = tmp[1].toFloat()
            }
            return o
        }

        fun update(fw: Int, fh: Int, left: Int, top: Int, side: Int, h: DoubleArray?, state: PanelTracker.State,
                   centre: IntArray?, th: PanelThermal?, outline: FloatArray?, frameNs: Long, k: DoubleArray,
                   gyroSign: Int) {
            val n = spec.rows * spec.cols
            st = St(fw, fh, left, top, side, h, state, centre, th?.cellC?.copyOf(), th?.cellDelta?.copyOf(),
                th?.let { t -> BooleanArray(n) { t.isHotspot(it) } }, outline, frameNs, k, gyroSign)
            postInvalidateOnAnimation()
        }

        /** Image motion since the camera frame [s] was analysed, from the gyro: lets the grid move at the display's
         *  rate (60 Hz) between 30 fps camera frames, and through motion-blurred ones. */
        private fun sinceFrame(s: St): DoubleArray? {
            if (s.gyroSign == 0 || s.h == null) return null
            val w = gyro.between(gyro.at(s.frameNs), gyro.now())
            val r = Mat3.rotation(w[0], w[1], w[2])
            val rr = if (s.gyroSign > 0) Mat3.transpose(r) else r
            return Mat3.mul(s.k, Mat3.mul(rr, Mat3.inv(s.k)))
        }

        override fun onDraw(canvas: Canvas) {
            val s0 = st ?: return
            val m = sinceFrame(s0)
            val s = if (m == null) s0 else St(s0.fw, s0.fh, s0.left, s0.top, s0.side, Mat3.mul(m, s0.h!!), s0.state,
                s0.centre, s0.temps, s0.deltas, s0.hot, s0.outline?.let { warp(m, it) }, s0.frameNs, s0.k, s0.gyroSign)
            if (s0.h != null) postInvalidateOnAnimation()
            val scale = minOf(width.toFloat() / s.fw, height.toFloat() / s.fh)
            val ox = (width - s.fw * scale) / 2
            val oy = (height - s.fh * scale) / 2
            fun sx(x: Double) = (ox + x * scale).toFloat()
            fun sy(y: Double) = (oy + y * scale).toFloat()
            canvas.drawRect(sx(s.left.toDouble()), sy(s.top.toDouble()), sx((s.left + s.side).toDouble()),
                sy((s.top + s.side).toDouble()), box)
            val h = s.h
            if (h != null) {
                fun pt(u: Double, v: Double): Pair<Float, Float>? {
                    val w = Mat3.apply(h, u, v, tmp)
                    return if (w > 0) sx(tmp[0]) to sy(tmp[1]) else null
                }
                for (r in 0 until spec.rows) for (c in 0 until spec.cols) {
                    val k = r * spec.cols + c
                    val fill = when {
                        s.hot?.get(k) == true -> if ((s.deltas?.get(k) ?: 0.0) > 0) hotFill else coldFill
                        s.centre != null && s.centre[0] == r && s.centre[1] == c -> centreFill
                        else -> null
                    } ?: continue
                    val q = listOf(pt(c.toDouble(), r.toDouble()), pt(c + 1.0, r.toDouble()),
                        pt(c + 1.0, r + 1.0), pt(c.toDouble(), r + 1.0))
                    if (q.any { it == null }) continue
                    val path = Path().apply {
                        moveTo(q[0]!!.first, q[0]!!.second)
                        for (i in 1 until 4) lineTo(q[i]!!.first, q[i]!!.second)
                        close()
                    }
                    canvas.drawPath(path, fill)
                }
                grid.color = if (s.state == PanelTracker.State.COASTING) Color.rgb(255, 165, 0) else Color.RED
                for (u in 0..spec.cols) {
                    val a = pt(u.toDouble(), 0.0); val b = pt(u.toDouble(), spec.rows.toDouble())
                    if (a != null && b != null) canvas.drawLine(a.first, a.second, b.first, b.second, grid)
                }
                for (v in 0..spec.rows) {
                    val a = pt(0.0, v.toDouble()); val b = pt(spec.cols.toDouble(), v.toDouble())
                    if (a != null && b != null) canvas.drawLine(a.first, a.second, b.first, b.second, grid)
                }
                for (r in 0 until spec.rows) for (c in 0 until spec.cols) {
                    val p = pt(c + 0.5, r + 0.5) ?: continue
                    val p1 = pt(c + 1.0, r + 0.5) ?: continue
                    val p0 = pt(c.toDouble(), r + 0.5) ?: continue
                    val cellPx = abs(p1.first - p0.first).coerceIn(30f, 600f)
                    label.textSize = cellPx * 0.22f
                    temp.textSize = cellPx * 0.2f
                    canvas.drawText("$r,$c", p.first - label.textSize, p.second, label)
                    val t = s.temps?.get(r * spec.cols + c)
                    if (t != null && !t.isNaN()) {
                        canvas.drawText(String.format(Locale.US, "%.1f°", t), p.first - temp.textSize * 1.3f,
                            p.second + temp.textSize * 1.2f, temp)
                    }
                }
            }
            s.outline?.let { o ->
                val path = Path()
                var pen = false
                for (i in 0 until o.size / 2) {
                    val x = o[2 * i]; val y = o[2 * i + 1]
                    if (x.isNaN() || abs(x) > 1e5 || abs(y) > 1e5) { pen = false; continue }
                    if (pen) path.lineTo(sx(x.toDouble()), sy(y.toDouble())) else path.moveTo(sx(x.toDouble()), sy(y.toDouble()))
                    pen = true
                }
                canvas.drawPath(path, view)
            }
            val cx = sx(s.fw / 2.0); val cy = sy(s.fh / 2.0)
            canvas.drawLine(cx - 30, cy, cx + 30, cy, cross)
            canvas.drawLine(cx, cy - 30, cx, cy + 30, cross)
        }
    }

    /** Rec: every analysed crop as JPEG plus a CSV row, <external files>/panel_debug/<time>/, for replay on a PC. */
    private class DebugRecorder(root: File) {
        private val dir = File(root, java.text.SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(java.util.Date()))
            .apply { mkdirs() }
        private val csv = File(dir, "frames.csv").bufferedWriter().apply {
            write("frame,sensor_ns,gyro_x,gyro_y,gyro_z,state,present,samples,fits,infer_ms,track_ms\n")
        }
        private var n = 0

        fun write(crop: Bitmap, sensorNs: Long, gyro: DoubleArray, state: PanelTracker.State, present: Float,
                  samples: Int, fits: String, inferMs: Long, trackMs: Long) {
            File(dir, String.format(Locale.US, "%06d.jpg", n)).outputStream().use {
                crop.compress(Bitmap.CompressFormat.JPEG, 92, it)
            }
            csv.write(String.format(Locale.US, "%d,%d,%.6f,%.6f,%.6f,%s,%.3f,%d,%s,%d,%d\n", n, sensorNs,
                gyro[0], gyro[1], gyro[2], state, present, samples, fits, inferMs, trackMs))
            n++
        }

        fun close() = csv.close()
    }

    companion object {
        private const val TAG = "SolarCells"
        private const val MODEL_ASSET = "panelseg.pte"
        private const val SMALL_ASSET = "panelseg_small.pte"
        private const val SMALL = 192
        private const val SMALL_THREADS = 4   // the A35's four big cores
        private const val BIG_THREADS = 2
        private const val TARGET_FPS = 30
        private const val BLUR_RAD_S = 1.5   // turning faster than ~85 deg/s smears the frame: follow the gyro
        private const val CALIB_ASSET = "thermal_calib.json"
        private const val SIZE = 384
        private const val BOX_FRACTION = 0.9f
        private const val PAIR_TOLERANCE_NS = 120_000_000L
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)

        fun assetFilePath(context: Context, name: String): String {
            val file = File(context.filesDir, name)
            context.assets.open(name).use { input -> file.outputStream().use { input.copyTo(it) } }
            return file.absolutePath
        }
    }
}
