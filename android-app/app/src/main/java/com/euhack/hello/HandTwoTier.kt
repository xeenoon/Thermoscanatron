package com.euhack.hello

import android.os.Process
import android.os.Debug
import android.os.SystemClock
import android.util.Log
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * The small HandSegNet runs every camera frame with four native threads. The big model is disabled by default;
 * diagnostic launches can enable it on a low-priority background thread whenever it is free.
 *
 * The outline always comes from the small model on the current frame: a hand moves on its own, so any mask from
 * an earlier frame is drawn where the hand used to be (moving it by the camera's motion does not help when the
 * camera is still and the hand is not). The big model only checks the small model's "is there a hand" answer,
 * and only while its own answer is fresh (<= [MAX_AGE] frames old):
 *  - small says hand, big is sure there is none (< [VETO]): no hand (the small model's false positives);
 *  - small is unsure (> [MAYBE]) and big is sure there is one (> [CONFIRM]): hand (side-on and blurred hands).
 *
 * ExecuTorch Android 1.5.1 uses a process-wide pool: loading the big module last resets the thread count
 * for BOTH models. A Java background thread does not isolate its native inference work.
 * Not thread-safe: call [run] from the analysis thread.
 */
class HandTwoTier(smallPath: String, bigPath: String?, private val size: Int, val smallSize: Int,
                  private val prevMaskInput: Boolean = true,
                  private val backgroundEnabled: Boolean = false,
                  smallThreads: Int = SMALL_THREADS, bigThreads: Int = BIG_THREADS,
                  private val diagnostics: Boolean = false) {
    class Result(val mask: FloatArray, val present: Float, val smallPresent: Float, val bigPresent: Float?,
                 val smallMs: Long, val bigMs: Long)

    private val small = Module.load(smallPath, Module.LOAD_MODE_FILE, smallThreads)
    // Do not load an idle big module: its load would still reset the small model's native thread pool.
    private val big = if (backgroundEnabled) Module.load(requireNotNull(bigPath), Module.LOAD_MODE_FILE, bigThreads) else null

    init {
        Log.i("HandPerf", "config background=$backgroundEnabled smallThreads=$smallThreads bigThreads=${if (big != null) bigThreads else 0} " +
            "smallSize=$smallSize bigSize=$size diagnostics=$diagnostics (thread pool is shared; last load wins)")
    }
    // Small model input: RGB plus (prevMaskInput) the previous frame's mask as a 4th channel, so it can carry a
    // thumb or a blurred hand over from the last frame instead of starting from nothing every frame.
    private val inputSmall = FloatArray((if (prevMaskInput) 4 else 3) * smallSize * smallSize)
    private val channels = if (prevMaskInput) 4L else 3L
    private val inputBig = if (big != null) FloatArray(3 * size * size) else null
    private val executor = if (big != null) Executors.newSingleThreadExecutor { r ->
        Thread({ Process.setThreadPriority(Process.THREAD_PRIORITY_BACKGROUND); r.run() }, "hand-big")
    } else null
    private val busy = AtomicBoolean(false)
    private var frame = 0L

    private class Big(val frame: Long, val present: Float)
    @Volatile private var latestBig: Big? = null
    @Volatile private var bigMs = 0L

    /**
     * One frame: [pixels] the smallSize x smallSize ARGB crop for the small model; [bigPixels] produces the
     * size x size crop, and is only called when the big model is free to take this frame.
     */
    fun run(pixels: IntArray, bigPixels: () -> IntArray, mean: FloatArray, std: FloatArray): Result {
        val f = frame++
        val tStart = SystemClock.elapsedRealtimeNanos()
        val bigBusy = busy.get()
        normalise(pixels, inputSmall, smallSize, mean, std)
        val tNormalise = SystemClock.elapsedRealtimeNanos()
        val value = EValue.from(Tensor.fromBlob(inputSmall, longArrayOf(1, channels, smallSize.toLong(), smallSize.toLong())))
        val tInput = SystemClock.elapsedRealtimeNanos()
        val cpuStart = if (diagnostics) Debug.threadCpuTimeNanos() else 0L
        val out = small.forward(value)
        val cpuNs = if (diagnostics) Debug.threadCpuTimeNanos() - cpuStart else 0L
        val tForward = SystemClock.elapsedRealtimeNanos()
        val smallMs = (tForward - tNormalise) / 1_000_000
        val mask = out[0].toTensor().dataAsFloatArray
        val pSmall = out[1].toTensor().dataAsFloatArray[0]
        if (prevMaskInput) {
            val plane = smallSize * smallSize
            for (i in 0 until plane) inputSmall[3 * plane + i] = if (mask[i] > 0.5f) 1f else 0f
        }

        val tOutput = SystemClock.elapsedRealtimeNanos()
        if (big != null && inputBig != null && executor != null && busy.compareAndSet(false, true)) {
            normalise(bigPixels(), inputBig, size, mean, std)
            executor.execute {
                try {
                    val t1 = SystemClock.elapsedRealtime()
                    val o = big.forward(EValue.from(Tensor.fromBlob(inputBig, longArrayOf(1, 3, size.toLong(), size.toLong()))))
                    latestBig = Big(f, o[1].toTensor().dataAsFloatArray[0])
                    bigMs = SystemClock.elapsedRealtime() - t1
                } catch (e: Exception) {
                    Log.e("HandTwoTier", "big model failed", e)
                } finally {
                    busy.set(false)
                }
            }
        }
        val tBigPrep = SystemClock.elapsedRealtimeNanos()
        if (diagnostics) Log.i("HandPerf", "frame=$f background=$backgroundEnabled bigBusyAtStart=$bigBusy " +
            "normaliseUs=${(tNormalise - tStart) / 1000} inputUs=${(tInput - tNormalise) / 1000} " +
            "forwardUs=${(tForward - tInput) / 1000} callerCpuUs=${cpuNs / 1000} " +
            "outputUs=${(tOutput - tForward) / 1000} bigPrepUs=${(tBigPrep - tOutput) / 1000}")

        val b = latestBig?.takeIf { f - it.frame <= MAX_AGE }
        val present = when {
            b == null -> pSmall
            pSmall > 0.5f && b.present < VETO -> b.present
            pSmall <= 0.5f && pSmall > MAYBE && b.present > CONFIRM -> b.present
            else -> pSmall
        }
        return Result(mask, present, pSmall, b?.present, smallMs, bigMs)
    }

    fun close() {
        executor?.execute { big?.destroy() }
        executor?.shutdown()
        small.destroy()
    }

    private fun normalise(px: IntArray, dst: FloatArray, n: Int, mean: FloatArray, std: FloatArray) {
        val plane = n * n
        for (i in 0 until plane) {
            val p = px[i]
            dst[i] = (((p shr 16) and 0xFF) / 255f - mean[0]) / std[0]
            dst[plane + i] = (((p shr 8) and 0xFF) / 255f - mean[1]) / std[1]
            dst[2 * plane + i] = ((p and 0xFF) / 255f - mean[2]) / std[2]
        }
    }

    companion object {
        const val MAX_AGE = 6        // frames: 0.2 s at 30 fps
        const val VETO = 0.2f
        const val MAYBE = 0.2f
        const val CONFIRM = 0.8f
        const val SMALL_THREADS = 4  // the A35's four big cores
        const val BIG_THREADS = 2
    }
}
