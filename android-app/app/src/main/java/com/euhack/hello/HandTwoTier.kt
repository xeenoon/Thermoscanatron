package com.euhack.hello

import android.os.Process
import android.os.SystemClock
import android.util.Log
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * HandSegNet as two models: the small one (same network at [smallSize] px, fine-tuned from the big one;
 * segkit-train --size 256 --prev-mask --init) on every camera frame, the big one ([size] px) on a low-priority background
 * thread whenever it is free.
 *
 * The outline always comes from the small model on the current frame: a hand moves on its own, so any mask from
 * an earlier frame is drawn where the hand used to be (moving it by the camera's motion does not help when the
 * camera is still and the hand is not). The big model only checks the small model's "is there a hand" answer,
 * and only while its own answer is fresh (<= [MAX_AGE] frames old):
 *  - small says hand, big is sure there is none (< [VETO]): no hand (the small model's false positives);
 *  - small is unsure (> [MAYBE]) and big is sure there is one (> [CONFIRM]): hand (side-on and blurred hands).
 *
 * The two models get separate thread budgets ([SMALL_THREADS], [BIG_THREADS]) so the background one cannot
 * starve the one the camera waits for. Not thread-safe: call [run] from the analysis thread.
 */
class HandTwoTier(smallPath: String, bigPath: String, private val size: Int, val smallSize: Int,
                  private val prevMaskInput: Boolean = true) {
    class Result(val mask: FloatArray, val present: Float, val smallPresent: Float, val bigPresent: Float?,
                 val smallMs: Long, val bigMs: Long)

    private val small = Module.load(smallPath, Module.LOAD_MODE_FILE, SMALL_THREADS)
    private val big = Module.load(bigPath, Module.LOAD_MODE_FILE, BIG_THREADS)
    // Small model input: RGB plus (prevMaskInput) the previous frame's mask as a 4th channel, so it can carry a
    // thumb or a blurred hand over from the last frame instead of starting from nothing every frame.
    private val inputSmall = FloatArray((if (prevMaskInput) 4 else 3) * smallSize * smallSize)
    private val channels = if (prevMaskInput) 4L else 3L
    private val inputBig = FloatArray(3 * size * size)
    private val executor = Executors.newSingleThreadExecutor { r ->
        Thread({ Process.setThreadPriority(Process.THREAD_PRIORITY_BACKGROUND); r.run() }, "hand-big")
    }
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
        normalise(pixels, inputSmall, smallSize, mean, std)
        val t0 = SystemClock.elapsedRealtime()
        val out = small.forward(EValue.from(Tensor.fromBlob(inputSmall, longArrayOf(1, channels, smallSize.toLong(), smallSize.toLong()))))
        val smallMs = SystemClock.elapsedRealtime() - t0
        val mask = out[0].toTensor().dataAsFloatArray
        val pSmall = out[1].toTensor().dataAsFloatArray[0]
        if (prevMaskInput) {
            val plane = smallSize * smallSize
            for (i in 0 until plane) inputSmall[3 * plane + i] = if (mask[i] > 0.5f) 1f else 0f
        }

        if (busy.compareAndSet(false, true)) {
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
        executor.execute { big.destroy() }
        executor.shutdown()
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
