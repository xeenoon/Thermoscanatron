package com.euhack.hello

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.view.Gravity
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.FrameLayout
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.File
import java.util.Locale
import java.util.concurrent.Executors

/**
 * Live hand demo: the centred square of the camera frame goes through HandSegNet (ExecuTorch, XNNPACK),
 * and the predicted outline is drawn in green over the preview when the model says a hand is present.
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
    private lateinit var logButton: Button

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
        logButton = Button(this).apply {
            text = "Log NO-HAND: off"
            setOnClickListener { toggleLogging() }
        }
        setContentView(FrameLayout(this).apply {
            setBackgroundColor(Color.BLACK)
            addView(previewView, FrameLayout.LayoutParams(-1, -1))
            addView(outlineView, FrameLayout.LayoutParams(-1, -1))
            addView(status, FrameLayout.LayoutParams(-1, -2, Gravity.TOP).apply { topMargin = 120 })
            addView(logButton, FrameLayout.LayoutParams(-2, -2, Gravity.BOTTOM or Gravity.CENTER_HORIZONTAL)
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

    private fun toggleLogging() {
        // Flip on the analysis thread so a frame never sees a half-started session.
        analysisExecutor.execute {
            val label = if (logger.enabled) {
                logger.stop()
                "Log NO-HAND: off (${logger.dumpCount} saved)"
            } else {
                val dir = logger.start()
                Log.i(TAG, "diagnostics -> $dir")
                "Log NO-HAND: ON"
            }
            runOnUiThread { logButton.text = label }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        analysisExecutor.execute { module?.destroy() }
        analysisExecutor.shutdown()
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()
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
            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyze(image: ImageProxy) {
        val net = module
        if (net == null) {
            image.close()
            return
        }
        // Upright frame, then the centred square (same box the overlay draws).
        val rotation = image.imageInfo.rotationDegrees
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
        val handVisible = present > PRESENT_THRESHOLD
        outlineView.update(frame.width, frame.height, left, top, side, if (handVisible) edgePoints(mask) else null)
        logger.onFrame(now, crop, input, mask, present, inferMs, handVisible, frame.width, frame.height, rotation)
        val logging = if (logger.enabled) "   |   ${logger.dumpCount} dumped" else ""
        runOnUiThread {
            status.text = String.format(
                Locale.US, "%s  %.2f   |   %d ms model   |   %.0f fps%s",
                if (handVisible) "HAND" else "NO HAND", present, inferMs, fps, logging,
            )
            status.setTextColor(if (handVisible) Color.GREEN else Color.WHITE)
        }
    }

    /** Mask pixels (prob > 0.5) with a 4-neighbour outside the mask, as (u, v) pairs in model pixels. */
    private fun edgePoints(mask: FloatArray): FloatArray {
        val out = ArrayList<Float>()
        fun inside(u: Int, v: Int) = u in 0 until SIZE && v in 0 until SIZE && mask[v * SIZE + u] > 0.5f
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
