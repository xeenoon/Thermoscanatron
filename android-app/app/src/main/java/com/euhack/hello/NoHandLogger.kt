package com.euhack.hello

import android.graphics.Bitmap
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors

/**
 * Diagnostics for the hand demo. While enabled:
 *  - frames.csv gets one line per frame (every frame, hand or not) so flicker and streaks are visible;
 *  - every frame the model calls NO HAND (rate-limited) is dumped as the exact network input
 *    (<n>_input.png, 384x384 RGB before normalisation), the predicted mask (<n>_mask.png, prob*255)
 *    and a line in nohand.jsonl with the scores;
 *  - the first [FLOAT_DUMPS] dumps also save the float tensor fed to the model (<n>_input_f32.bin,
 *    little-endian CHW), to check the phone's preprocessing against Python bit for bit.
 * Files land in <external files>/diagnostics/session_<time>/.
 */
class NoHandLogger(private val root: File) {
    private val io = Executors.newSingleThreadExecutor()
    private var dir: File? = null
    private var frameIndex = 0L
    private var dumps = 0
    private var lastDumpMs = 0L

    val enabled get() = dir != null
    val dumpCount get() = dumps

    fun start(): File {
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val d = File(root, "diagnostics/session_$stamp").apply { mkdirs() }
        File(d, "frames.csv").writeText("frame,ms,present,mask_frac,mask_max,infer_ms,brightness\n")
        dir = d
        frameIndex = 0
        dumps = 0
        lastDumpMs = 0
        return d
    }

    fun stop() {
        dir = null
    }

    /** Called on the analysis thread after every inference. [crop] must not be reused by the caller. */
    fun onFrame(nowMs: Long, crop: Bitmap, input: FloatArray, mask: FloatArray, present: Float, inferMs: Long,
                handVisible: Boolean, frameW: Int, frameH: Int, rotation: Int) {
        val d = dir ?: return
        val index = frameIndex++
        var above = 0
        var maxProb = 0f
        for (p in mask) {
            if (p > 0.5f) above++
            if (p > maxProb) maxProb = p
        }
        val maskFrac = above.toFloat() / mask.size
        val brightness = meanBrightness(crop)
        val csvLine = String.format(Locale.US, "%d,%d,%.4f,%.4f,%.4f,%d,%.1f\n",
            index, nowMs, present, maskFrac, maxProb, inferMs, brightness)

        val dump = !handVisible && nowMs - lastDumpMs >= MIN_DUMP_INTERVAL_MS
        if (dump) lastDumpMs = nowMs
        val dumpNo = if (dump) dumps++ else -1
        val maskCopy = if (dump) mask.copyOf() else null
        val inputCopy = if (dump && dumpNo < FLOAT_DUMPS) input.copyOf() else null

        io.execute {
            try {
                File(d, "frames.csv").appendText(csvLine)
                if (maskCopy != null) {
                    val name = String.format(Locale.US, "%05d", dumpNo)
                    savePng(crop, File(d, "${name}_input.png"))
                    savePng(maskBitmap(maskCopy), File(d, "${name}_mask.png"))
                    inputCopy?.let { saveFloats(it, File(d, "${name}_input_f32.bin")) }
                    File(d, "nohand.jsonl").appendText(String.format(Locale.US,
                        "{\"dump\":%d,\"frame\":%d,\"ms\":%d,\"present\":%.5f,\"mask_frac\":%.5f,\"mask_max\":%.5f," +
                            "\"infer_ms\":%d,\"brightness\":%.1f,\"frame_w\":%d,\"frame_h\":%d,\"rotation\":%d}\n",
                        dumpNo, index, nowMs, present, maskFrac, maxProb, inferMs, brightness, frameW, frameH, rotation))
                }
            } catch (e: Exception) {
                Log.e("NoHandLogger", "write failed", e)
            }
        }
    }

    private fun meanBrightness(bmp: Bitmap): Float {
        val px = IntArray(bmp.width * bmp.height)
        bmp.getPixels(px, 0, bmp.width, 0, 0, bmp.width, bmp.height)
        var sum = 0L
        for (p in px) sum += ((p shr 16) and 0xFF) + ((p shr 8) and 0xFF) + (p and 0xFF)
        return sum / (3f * px.size)
    }

    private fun maskBitmap(mask: FloatArray): Bitmap {
        val px = IntArray(mask.size) {
            val v = (mask[it].coerceIn(0f, 1f) * 255).toInt()
            (0xFF shl 24) or (v shl 16) or (v shl 8) or v
        }
        return Bitmap.createBitmap(px, SIZE, SIZE, Bitmap.Config.ARGB_8888)
    }

    private fun savePng(bmp: Bitmap, file: File) {
        FileOutputStream(file).use { bmp.compress(Bitmap.CompressFormat.PNG, 100, it) }
    }

    private fun saveFloats(values: FloatArray, file: File) {
        val buf = ByteBuffer.allocate(values.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        buf.asFloatBuffer().put(values)
        file.writeBytes(buf.array())
    }

    companion object {
        private const val SIZE = 384
        private const val MIN_DUMP_INTERVAL_MS = 300L
        private const val FLOAT_DUMPS = 3
    }
}
