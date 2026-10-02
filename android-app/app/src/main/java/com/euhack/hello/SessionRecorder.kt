package com.euhack.hello

import android.graphics.Bitmap
import android.util.Log
import org.json.JSONObject
import java.io.BufferedOutputStream
import java.io.BufferedWriter
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale
import java.util.concurrent.Executors

/**
 * Records the camera and the thermal camera together, for fitting the camera-to-thermal alignment.
 * One folder per recording, <external files>/sessions/<time>/:
 *  - meta.json: camera characteristics, analysis-stream geometry, file formats, counts;
 *  - frames.csv: one row per analysed camera frame (sensor timestamp, crop box, model scores);
 *  - crops/<frame>.jpg: the 384x384 model input (upright RGB crop of the frame);
 *  - masks/<frame>.png: the model's hand probability for that crop, *255 (grey);
 *  - thermal.bin: every thermal packet as [int64 LE elapsedRealtimeNanos at arrival][1566-byte THM2 packet].
 * Camera timestamps are ImageInfo.timestamp (see meta "timestamp_source"); "analysis_ns" is
 * elapsedRealtimeNanos when the frame was analysed, the same clock as the thermal arrival times.
 * All file writes happen on one background thread.
 */
class SessionRecorder(private val root: File) {
    private val io = Executors.newSingleThreadExecutor()
    @Volatile private var session: Session? = null

    private class Session(val dir: File, val meta: JSONObject) {
        val frames: BufferedWriter = File(dir, "frames.csv").bufferedWriter()
        val thermal = BufferedOutputStream(FileOutputStream(File(dir, "thermal.bin")))
        @Volatile var cameraFrames = 0L
        @Volatile var thermalPackets = 0L
        var geometryWritten = false
    }

    val active get() = session != null
    val cameraFrames get() = session?.cameraFrames ?: 0L
    val thermalPackets get() = session?.thermalPackets ?: 0L

    fun start(stamp: String, meta: JSONObject): File {
        val dir = File(root, "sessions/$stamp").apply {
            mkdirs()
            File(this, "crops").mkdirs()
            File(this, "masks").mkdirs()
        }
        meta.put("thermal_format", "repeated [int64 LE elapsedRealtimeNanos][THM2 packet, 1566 bytes]")
        val s = Session(dir, meta)
        io.execute {
            s.frames.write("frame,sensor_ts_ns,analysis_ns,frame_w,frame_h,rotation,box_left,box_top,box_side," +
                "present,hand_visible,mask_frac,infer_ms\n")
            writeMeta(s)
        }
        session = s
        return dir
    }

    fun stop() {
        val s = session ?: return
        session = null
        io.execute {
            try {
                s.frames.close()
                s.thermal.close()
                s.meta.put("camera_frames", s.cameraFrames)
                s.meta.put("thermal_packets", s.thermalPackets)
                writeMeta(s)
            } catch (e: Exception) {
                Log.e(TAG, "closing session failed", e)
            }
        }
    }

    /**
     * Called on the analysis thread. [crop] must not be reused by the caller; [mask] is copied here.
     * [sensorToBuffer] maps sensor active-array pixels to analysis-buffer pixels (row-major 3x3).
     */
    fun onCameraFrame(sensorTsNs: Long, analysisNs: Long, frameW: Int, frameH: Int, rotation: Int,
                      left: Int, top: Int, side: Int, crop: Bitmap, mask: FloatArray, present: Float,
                      handVisible: Boolean, inferMs: Long, sensorToBuffer: FloatArray, bufferW: Int, bufferH: Int) {
        val s = session ?: return
        val index = s.cameraFrames++
        var above = 0
        val grey = IntArray(mask.size) {
            val p = mask[it]
            if (p > 0.5f) above++
            val v = (p.coerceIn(0f, 1f) * 255).toInt()
            (0xFF shl 24) or (v shl 16) or (v shl 8) or v
        }
        val maskFrac = above.toFloat() / mask.size
        io.execute {
            try {
                if (!s.geometryWritten) {
                    s.geometryWritten = true
                    s.meta.put("analysis", JSONObject()
                        .put("buffer_w", bufferW).put("buffer_h", bufferH)
                        .put("rotation_degrees", rotation)
                        .put("upright_w", frameW).put("upright_h", frameH)
                        .put("box_left", left).put("box_top", top).put("box_side", side)
                        .put("model_size", crop.width)
                        .put("sensor_to_buffer", org.json.JSONArray(sensorToBuffer.map { it.toDouble() })))
                    writeMeta(s)
                }
                val name = String.format(Locale.US, "%06d", index)
                FileOutputStream(File(s.dir, "crops/$name.jpg")).use { crop.compress(Bitmap.CompressFormat.JPEG, 92, it) }
                val maskBmp = Bitmap.createBitmap(grey, crop.width, crop.height, Bitmap.Config.ARGB_8888)
                FileOutputStream(File(s.dir, "masks/$name.png")).use { maskBmp.compress(Bitmap.CompressFormat.PNG, 100, it) }
                s.frames.write(String.format(Locale.US, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%.5f,%d,%.5f,%d\n",
                    index, sensorTsNs, analysisNs, frameW, frameH, rotation, left, top, side,
                    present, if (handVisible) 1 else 0, maskFrac, inferMs))
            } catch (e: Exception) {
                Log.e(TAG, "camera frame write failed", e)
            }
        }
    }

    /** Called on the USB thread. */
    fun onThermalFrame(frame: ThermalFrame) {
        val s = session ?: return
        s.thermalPackets++
        io.execute {
            try {
                s.thermal.write(ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(frame.receivedNs).array())
                s.thermal.write(frame.packet)
            } catch (e: Exception) {
                Log.e(TAG, "thermal write failed", e)
            }
        }
    }

    private fun writeMeta(s: Session) {
        File(s.dir, "meta.json").writeText(s.meta.toString(2))
    }

    companion object {
        private const val TAG = "SessionRecorder"
    }
}
