package com.euhack.hello

import android.graphics.Bitmap
import android.graphics.Color
import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.sqrt

/**
 * Camera + thermal fusion for the calibrated rig. Every camera pixel is mapped into the thermal image
 * through the calibration ([ThermalPose], at the depth of the hand, or a default scene depth); pixels the
 * thermal camera does not see are black. The 32x24 temperatures are upsampled to camera resolution with a
 * guided filter that uses the camera image as the guide, so temperature edges snap to the object edges the
 * phone sees:
 *   K. He, J. Sun, X. Tang, "Guided Image Filtering", IEEE TPAMI 35(6) (2013) 1397-1409,
 *   doi:10.1109/TPAMI.2012.213.
 * (the same job joint bilateral upsampling does: J. Kopf, M. F. Cohen, D. Lischinski, M. Uyttendaele,
 * "Joint Bilateral Upsampling", ACM TOG 26(3) (2007) 96, doi:10.1145/1276377.1276497; the guided filter is
 * used here because it runs in O(pixels) with box filters.)
 * Before that the thermal stream is denoised like a thermal camera's: a 3x3 binomial blur (also hides the
 * chess-pattern seam between the two subpages) and a per-pixel running average that follows real changes
 * (> 1.5 C) quickly. The colour range follows the scene's 1st-99th percentile slowly, never narrower than
 * [MIN_SPAN_C], so a flat scene shows real structure instead of amplified sensor noise.
 * The result is colour-mapped (ironbow), blended with the camera image and its edges drawn in, so each
 * temperature can be read off the object it belongs to. Like Hong et al. (2022) (see
 * [ThermalCalibration]), the camera supplies the geometry and the thermal camera the temperatures: the
 * guided filter sharpens where temperatures change, it does not add temperature resolution.
 *
 * Not thread-safe: call [render] from one thread.
 */
class FusionRenderer(val width: Int = 240, val height: Int = 320) {
    private val n = width * height
    private val mapU = FloatArray(n)
    private val mapV = FloatArray(n)
    private val inside = BooleanArray(n)
    private var mapPose: ThermalPose? = null
    private var mapDepth = 0.0
    private var mapFrameW = 0
    private var mapFrameH = 0

    private val rgb = IntArray(n)
    private val guide = FloatArray(n)
    private val p = FloatArray(n)
    private val q = FloatArray(n)
    private val out = IntArray(n)
    private val meanI = FloatArray(n)
    private val meanP = FloatArray(n)
    private val corrII = FloatArray(n)
    private val corrIP = FloatArray(n)
    private val a = FloatArray(n)
    private val b = FloatArray(n)
    private val integral = DoubleArray((width + 1) * (height + 1))

    /** Output; read it under synchronized(bitmap). */
    val bitmap: Bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

    class Stats(val lowC: Float, val highC: Float, val handC: Float?, val overlap: Float)

    private val smooth = FloatArray(ThermalGeometry.W * ThermalGeometry.H)
    private val blurTmp = FloatArray(ThermalGeometry.W * ThermalGeometry.H)
    private val blurOut = FloatArray(ThermalGeometry.W * ThermalGeometry.H)
    private var lastThermal: ThermalFrame? = null
    private var lowC = Float.NaN
    private var highC = Float.NaN

    /** Folds a new thermal frame into the running average and the colour range (once per frame). */
    private fun updateThermal(frame: ThermalFrame) {
        if (frame === lastThermal) return
        val first = lastThermal == null
        lastThermal = frame
        val w = ThermalGeometry.W
        val h = ThermalGeometry.H
        val t = frame.celsius
        fun at(x: Int, y: Int) = t[y.coerceIn(0, h - 1) * w + x.coerceIn(0, w - 1)]
        for (y in 0 until h) for (x in 0 until w) blurTmp[y * w + x] = (at(x - 1, y) + 2 * at(x, y) + at(x + 1, y)) / 4
        for (y in 0 until h) for (x in 0 until w) {
            val up = blurTmp[(y - 1).coerceAtLeast(0) * w + x]
            val down = blurTmp[(y + 1).coerceAtMost(h - 1) * w + x]
            blurOut[y * w + x] = (up + 2 * blurTmp[y * w + x] + down) / 4
        }
        for (i in smooth.indices) {
            val d = blurOut[i] - smooth[i]
            smooth[i] = if (first) blurOut[i] else smooth[i] + d * (if (abs(d) > FAST_DELTA_C) 0.8f else 0.3f)
        }
        val sorted = smooth.sortedArray()
        var lo = sorted[(sorted.size * 0.01f).toInt()]
        var hi = sorted[(sorted.size * 0.99f).toInt()]
        if (hi - lo < MIN_SPAN_C) {
            val c = (hi + lo) / 2
            lo = c - MIN_SPAN_C / 2
            hi = c + MIN_SPAN_C / 2
        }
        if (lowC.isNaN()) {
            lowC = lo; highC = hi
        } else {
            lowC += (lo - lowC) * RANGE_RATE
            highC += (hi - highC) * RANGE_RATE
        }
    }

    /**
     * [frame] is the upright camera frame; [hand] (optional) the model's hand probabilities over the crop
     * box ([maskSize]^2 at [left], [top], [side] in frame pixels). [depthCm] is where the mapping is exact.
     */
    fun render(frame: Bitmap, thermal: ThermalFrame, pose: ThermalPose, cam: CameraIntrinsics, depthCm: Double,
               hand: FloatArray?, maskSize: Int, left: Int, top: Int, side: Int): Stats {
        updateMap(pose, cam, depthCm, frame.width, frame.height)
        updateThermal(thermal)
        Bitmap.createScaledBitmap(frame, width, height, true).getPixels(rgb, 0, width, 0, 0, width, height)

        val celsius = smooth
        val lo = lowC
        val hi = highC
        val span = hi - lo
        var seen = 0
        for (i in 0 until n) {
            val c = rgb[i]
            guide[i] = (0.299f * ((c shr 16) and 0xFF) + 0.587f * ((c shr 8) and 0xFF) + 0.114f * (c and 0xFF)) / 255f
            p[i] = (sampleThermal(celsius, mapU[i], mapV[i]) - lo) / span
            if (inside[i]) seen++
        }
        guidedFilter(RADIUS, EPS)

        // Hand temperature: median of the (denoised) temperatures under the hand mask, unclipped.
        var handC: Float? = null
        if (hand != null && side > 0) {
            val sx = frame.width.toFloat() / width
            val sy = frame.height.toFloat() / height
            val temps = ArrayList<Float>()
            for (y in 0 until height) for (x in 0 until width) {
                val i = y * width + x
                if (!inside[i]) continue
                val mx = (((x + 0.5f) * sx - left) / side * maskSize).toInt()
                val my = (((y + 0.5f) * sy - top) / side * maskSize).toInt()
                if (mx !in 0 until maskSize || my !in 0 until maskSize) continue
                if (hand[my * maskSize + mx] > 0.5f) temps.add(sampleThermal(celsius, mapU[i], mapV[i]))
            }
            if (temps.size > 20) handC = temps.sorted()[temps.size / 2]
        }

        for (y in 0 until height) for (x in 0 until width) {
            val i = y * width + x
            if (!inside[i]) {
                out[i] = Color.BLACK
                continue
            }
            val base = heatColor(q[i])
            // Blend in the camera image so cold (dark) areas still show their objects, and draw its edges.
            val cam255 = guide[i] * 255 * CAMERA_BLEND
            val e = (edge(x, y) * 3f).coerceIn(0f, 1f) * 0.45f
            val r = (((base shr 16) and 0xFF) * 0.85f + cam255).coerceAtMost(255f) * (1 - e) + 255 * e
            val g = (((base shr 8) and 0xFF) * 0.85f + cam255).coerceAtMost(255f) * (1 - e) + 255 * e
            val bl = ((base and 0xFF) * 0.85f + cam255).coerceAtMost(255f) * (1 - e) + 255 * e
            out[i] = Color.rgb(r.toInt(), g.toInt(), bl.toInt())
        }
        synchronized(bitmap) { bitmap.setPixels(out, 0, width, 0, 0, width, height) }
        return Stats(lo, hi, handC, seen.toFloat() / n)
    }

    /** Camera pixel -> thermal pixel, recomputed only when the pose, frame size or depth (>10%) changes. */
    private fun updateMap(pose: ThermalPose, cam: CameraIntrinsics, depthCm: Double, frameW: Int, frameH: Int) {
        if (pose === mapPose && frameW == mapFrameW && frameH == mapFrameH && abs(ln(depthCm / mapDepth)) < 0.1) return
        val sx = frameW.toDouble() / width
        val sy = frameH.toDouble() / height
        val uv = DoubleArray(2)
        for (y in 0 until height) for (x in 0 until width) {
            val fx = (x + 0.5) * sx - 0.5
            val fy = (y + 0.5) * sy - 0.5
            pose.project((fx - cam.cx) / cam.f * depthCm, (fy - cam.cy) / cam.f * depthCm, depthCm, uv)
            val i = y * width + x
            mapU[i] = uv[0].toFloat()
            mapV[i] = uv[1].toFloat()
            inside[i] = uv[0] >= -0.5 && uv[0] < ThermalGeometry.W - 0.5 && uv[1] >= -0.5 && uv[1] < ThermalGeometry.H - 0.5
        }
        mapPose = pose
        mapDepth = depthCm
        mapFrameW = frameW
        mapFrameH = frameH
    }

    private fun sampleThermal(t: FloatArray, u: Float, v: Float): Float {
        val x = u.coerceIn(0f, ThermalGeometry.W - 1.001f)
        val y = v.coerceIn(0f, ThermalGeometry.H - 1.001f)
        val x0 = x.toInt()
        val y0 = y.toInt()
        val fx = x - x0
        val fy = y - y0
        val i = y0 * ThermalGeometry.W + x0
        return t[i] * (1 - fx) * (1 - fy) + t[i + 1] * fx * (1 - fy) +
            t[i + ThermalGeometry.W] * (1 - fx) * fy + t[i + ThermalGeometry.W + 1] * fx * fy
    }

    /** He et al. guided filter: q = mean(a) * I + mean(b), a = cov(I, p) / (var(I) + eps), b = mean(p) - a mean(I). */
    private fun guidedFilter(r: Int, eps: Float) {
        boxMean(guide, meanI, r)
        boxMean(p, meanP, r)
        for (i in 0 until n) a[i] = guide[i] * guide[i]
        boxMean(a, corrII, r)
        for (i in 0 until n) a[i] = guide[i] * p[i]
        boxMean(a, corrIP, r)
        for (i in 0 until n) {
            val varI = corrII[i] - meanI[i] * meanI[i]
            val cov = corrIP[i] - meanI[i] * meanP[i]
            a[i] = cov / (varI + eps)
            b[i] = meanP[i] - a[i] * meanI[i]
        }
        boxMean(a, corrII, r)      // reuse buffers: mean a
        boxMean(b, corrIP, r)      // mean b
        for (i in 0 until n) q[i] = (corrII[i] * guide[i] + corrIP[i]).coerceIn(0f, 1f)
    }

    /** Mean over a (2r+1)^2 window, clipped at the borders, via an integral image. */
    private fun boxMean(src: FloatArray, dst: FloatArray, r: Int) {
        val w1 = width + 1
        java.util.Arrays.fill(integral, 0, w1, 0.0)
        for (y in 0 until height) {
            var row = 0.0
            integral[(y + 1) * w1] = 0.0
            for (x in 0 until width) {
                row += src[y * width + x]
                integral[(y + 1) * w1 + x + 1] = integral[y * w1 + x + 1] + row
            }
        }
        for (y in 0 until height) {
            val y0 = maxOf(0, y - r)
            val y1 = minOf(height, y + r + 1)
            for (x in 0 until width) {
                val x0 = maxOf(0, x - r)
                val x1 = minOf(width, x + r + 1)
                val s = integral[y1 * w1 + x1] - integral[y0 * w1 + x1] - integral[y1 * w1 + x0] + integral[y0 * w1 + x0]
                dst[y * width + x] = (s / ((y1 - y0) * (x1 - x0))).toFloat()
            }
        }
    }

    private fun edge(x: Int, y: Int): Float {
        if (x == 0 || y == 0 || x == width - 1 || y == height - 1) return 0f
        val i = y * width + x
        val gx = guide[i + 1] - guide[i - 1]
        val gy = guide[i + width] - guide[i - width]
        return sqrt(gx * gx + gy * gy)
    }

    private fun heatColor(value: Float): Int {
        val scaled = value.coerceIn(0f, 0.9999f) * (STOPS.size - 1)
        val i = scaled.toInt()
        val mix = scaled - i
        val c0 = STOPS[i]
        val c1 = STOPS[i + 1]
        return Color.rgb((c0[0] + (c1[0] - c0[0]) * mix).toInt(), (c0[1] + (c1[1] - c0[1]) * mix).toInt(),
            (c0[2] + (c1[2] - c0[2]) * mix).toInt())
    }

    companion object {
        /** Window radius in output pixels: about one thermal pixel's footprint (~7 px at 240 wide). */
        const val RADIUS = 8
        /**
         * Edge-preservation: smaller transfers more camera edges into the temperatures, but also turns sensor
         * noise into blotches shaped like the camera's edges; 0.01 keeps object edges without that.
         */
        const val EPS = 0.01f
        const val DEFAULT_DEPTH_CM = 100.0
        const val MIN_SPAN_C = 5f            // narrowest colour range: ~10x the sensor's frame-to-frame noise
        const val FAST_DELTA_C = 1.5f        // larger changes are real (motion), follow them quickly
        const val RANGE_RATE = 0.25f         // colour range follows the scene over ~4 thermal frames
        const val CAMERA_BLEND = 0.3f

        /** Ironbow: black -> indigo -> magenta -> red -> orange -> yellow -> white. */
        private val STOPS = arrayOf(
            intArrayOf(0, 0, 0), intArrayOf(30, 0, 110), intArrayOf(120, 0, 150), intArrayOf(200, 30, 90),
            intArrayOf(240, 90, 20), intArrayOf(255, 170, 0), intArrayOf(255, 235, 90), intArrayOf(255, 255, 255),
        )
    }
}
