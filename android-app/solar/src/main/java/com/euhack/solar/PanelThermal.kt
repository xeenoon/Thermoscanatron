package com.euhack.solar

import com.euhack.hello.CameraIntrinsics
import com.euhack.hello.ThermalGeometry
import com.euhack.hello.ThermalPose
import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.sqrt

/**
 * Per-cell panel temperatures from the thermal camera. Port of ML/src/segkit/panel/thermal.py.
 *
 * For a thermal frame and the panel homography of the camera frame it belongs to (analysis-frame pixels):
 * the panel plane comes from H = K [r1 r2 t] diag(cell w, cell h); every thermal pixel's footprint (3 x 3 rays
 * over +-0.75 px: the pixel plus the optics' blur) is cast from the thermal camera onto that plane. A pixel
 * whose whole footprint is inside one cell is a clean reading of it; one inside the cell area counts towards
 * the panel average. Cell temperature = median of its clean pixels.
 *
 * Over time each cell keeps a running average of its offset from the panel average of the same frame (so the
 * whole panel warming up does not fake a hotspot), and the panel average its own running average.
 * Hotspot: a cell more than [HOTSPOT_DELTA_C] above or below the panel average.
 */
class PanelThermal(private val spec: PanelSpec, private val pose: ThermalPose) {
    private val w = ThermalGeometry.W
    private val h = ThermalGeometry.H
    /** Footprint rays, camera coordinates: [pixel][ray][xyz]. */
    private val rays = Array(w * h) { p ->
        Array(9) { k ->
            val out = DoubleArray(3)
            pose.ray(p % w + OFFS[k % 3], p / w + OFFS[k / 3], out)
            out
        }
    }
    /** Outline of the thermal view (perimeter rays), for drawing where the thermal camera looks. */
    private val border: List<DoubleArray> = buildList {
        for (u in 0..w) add(DoubleArray(3).also { pose.ray(u - 0.5, -0.5, it) })
        for (v in 0..h) add(DoubleArray(3).also { pose.ray(w - 0.5, v - 0.5, it) })
        for (u in w downTo 0) add(DoubleArray(3).also { pose.ray(u - 0.5, h - 0.5, it) })
        for (v in h downTo 0) add(DoubleArray(3).also { pose.ray(-0.5, v - 0.5, it) })
    }

    val cellC = DoubleArray(spec.rows * spec.cols) { Double.NaN }      // smoothed cell temperature
    val cellDelta = DoubleArray(spec.rows * spec.cols) { Double.NaN }  // smoothed offset from the panel average
    val cellFrames = IntArray(spec.rows * spec.cols)
    var panelC = Double.NaN
        private set
    var lastPanelPixels = 0
        private set

    fun reset() {
        cellC.fill(Double.NaN)
        cellDelta.fill(Double.NaN)
        cellFrames.fill(0)
        panelC = Double.NaN
    }

    fun isHotspot(k: Int) = !cellDelta[k].isNaN() && abs(cellDelta[k]) > HOTSPOT_DELTA_C

    /** Panel plane in camera coordinates (cm): origin t, unit column axis r1, unit row axis r2; null if degenerate. */
    class Plane(val t: DoubleArray, val r1: DoubleArray, val r2: DoubleArray) {
        val n = doubleArrayOf(r1[1] * r2[2] - r1[2] * r2[1], r1[2] * r2[0] - r1[0] * r2[2], r1[0] * r2[1] - r1[1] * r2[0])
    }

    fun plane(hFrame: DoubleArray, cam: CameraIntrinsics): Plane? {
        val kInv = doubleArrayOf(1 / cam.f, 0.0, -cam.cx / cam.f, 0.0, 1 / cam.f, -cam.cy / cam.f, 0.0, 0.0, 1.0)
        var b = Mat3.mul(kInv, hFrame)
        if (b[8] < 0) b = DoubleArray(9) { -b[it] }
        val n1 = sqrt(b[0] * b[0] + b[3] * b[3] + b[6] * b[6])
        val n2 = sqrt(b[1] * b[1] + b[4] * b[4] + b[7] * b[7])
        if (n1 < 1e-12 || n2 < 1e-12) return null
        val lam = n1 / spec.cellWcm
        return Plane(
            doubleArrayOf(b[2] / lam, b[5] / lam, b[8] / lam),
            doubleArrayOf(b[0] / n1, b[3] / n1, b[6] / n1),
            doubleArrayOf(b[1] / n2, b[4] / n2, b[7] / n2),
        )
    }

    /** Panel (u, v) in cells where [ray] from the thermal camera meets the plane, into out; false if behind. */
    private fun cast(ray: DoubleArray, p: Plane, out: DoubleArray): Boolean {
        val c0 = pose.x; val c1 = pose.y; val c2 = pose.z
        val den = ray[0] * p.n[0] + ray[1] * p.n[1] + ray[2] * p.n[2]
        if (abs(den) < 1e-9) return false
        val s = ((p.t[0] - c0) * p.n[0] + (p.t[1] - c1) * p.n[1] + (p.t[2] - c2) * p.n[2]) / den
        if (s <= 0) return false
        val q0 = c0 + s * ray[0] - p.t[0]
        val q1 = c1 + s * ray[1] - p.t[1]
        val q2 = c2 + s * ray[2] - p.t[2]
        out[0] = (q0 * p.r1[0] + q1 * p.r1[1] + q2 * p.r1[2]) / spec.cellWcm
        out[1] = (q0 * p.r2[0] + q1 * p.r2[1] + q2 * p.r2[2]) / spec.cellHcm
        return true
    }

    /** Fold one thermal frame (celsius, row-major 32x24) seen with panel plane [p] into the running readings. */
    fun update(celsius: FloatArray, p: Plane) {
        val cells = Array(spec.rows * spec.cols) { ArrayList<Float>() }
        var sum = 0.0
        var count = 0
        val uv = DoubleArray(2)
        val m = EDGE_MARGIN
        for (px in 0 until w * h) {
            var onPanel = true
            var same = true
            var cu = -1
            var cv = -1
            for (k in 0 until 9) {
                if (!cast(rays[px][k], p, uv)) { onPanel = false; break }
                val u = uv[0]; val v = uv[1]
                if (u < m || u > spec.cols - m || v < m || v > spec.rows - m) { onPanel = false; break }
                val iu = floor(u).toInt(); val iv = floor(v).toInt()
                val fu = u - iu; val fv = v - iv
                if (k == 0) { cu = iu; cv = iv }
                if (iu != cu || iv != cv || fu < m || fu > 1 - m || fv < m || fv > 1 - m) same = false
            }
            if (!onPanel) continue
            sum += celsius[px]
            count++
            if (same) cells[cv * spec.cols + cu].add(celsius[px])
        }
        lastPanelPixels = count
        if (count < MIN_PANEL_PX) return
        val panel = sum / count
        panelC = if (panelC.isNaN()) panel else panelC + SMOOTH * (panel - panelC)
        for (k in cells.indices) {
            val vals = cells[k]
            if (vals.isEmpty()) continue
            vals.sort()
            val med = vals[vals.size / 2].toDouble()
            val d = med - panel
            cellDelta[k] = if (cellDelta[k].isNaN()) d else cellDelta[k] + SMOOTH * (d - cellDelta[k])
            cellC[k] = if (cellC[k].isNaN()) med else cellC[k] + SMOOTH * (med - cellC[k])
            cellFrames[k]++
        }
    }

    /** Thermal view outline on the panel plane, as analysis-frame pixels (x, y pairs; NaN where it misses). */
    fun viewOutline(p: Plane, cam: CameraIntrinsics): FloatArray {
        val out = FloatArray(border.size * 2)
        val uv = DoubleArray(2)
        for ((i, r) in border.withIndex()) {
            if (!cast(r, p, uv)) { out[2 * i] = Float.NaN; out[2 * i + 1] = Float.NaN; continue }
            // Back to a camera point, then through the pinhole.
            val x = p.t[0] + uv[0] * spec.cellWcm * p.r1[0] + uv[1] * spec.cellHcm * p.r2[0]
            val y = p.t[1] + uv[0] * spec.cellWcm * p.r1[1] + uv[1] * spec.cellHcm * p.r2[1]
            val z = p.t[2] + uv[0] * spec.cellWcm * p.r1[2] + uv[1] * spec.cellHcm * p.r2[2]
            out[2 * i] = (cam.f * x / z + cam.cx).toFloat()
            out[2 * i + 1] = (cam.f * y / z + cam.cy).toFloat()
        }
        return out
    }

    companion object {
        const val HOTSPOT_DELTA_C = 5.0
        const val EDGE_MARGIN = 0.05
        const val MIN_PANEL_PX = 6
        const val SMOOTH = 0.3
        private val OFFS = doubleArrayOf(-0.75, 0.0, 0.75)
    }
}
