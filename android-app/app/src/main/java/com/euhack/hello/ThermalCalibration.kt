package com.euhack.hello

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Finds where the thermal camera is relative to the phone camera from a few seconds of a hand seen by both,
 * with no assumptions about the mounting (any roll, large yaw/pitch, tens of cm of offset, mirrored or not).
 * Kotlin port of ML/src/segkit/thermal_calib.py (see there for the model); pure Kotlin so it is unit-tested
 * on the JVM against the same synthetic poses and the recorded session.
 *
 * Registration idea after Hong et al. (2022): the RGB image carries the geometry (the hand mask), the
 * thermal image the temperatures.
 *   F. Hong, J. Song, H. Meng, R. Wang, F. Fang, G. Zhang, "A novel framework on intelligent detection for
 *   module defects of PV plant combining the visible and infrared images", Solar Energy 236 (2022) 406-416,
 *   doi:10.1016/j.solener.2022.03.018.
 *
 *  0. Latency (thermal packet arrival vs camera capture) from geometry-free signals: the hand's size in the
 *     camera and the warm area in the thermal image rise and fall together as the hand moves nearer and
 *     further, whatever the mounting; their cross-correlation peak is the lag. It has to be measured this
 *     way: fitted together with the pose, a lag on a hand moving in circles is indistinguishable from a
 *     roll of the sensor (a delayed circle is a rotated circle), which once gave roll 41 deg instead of 95.
 *  1. Global pose from points: hand centroid at its estimated depth (3D, camera) vs warm-blob centroid
 *     (2D, thermal); robust Levenberg-Marquardt from a grid of starting rotations, both mirror hypotheses.
 *  2. Silhouette refinement: Nelder-Mead on the correlation between each thermal pixel's predicted hand
 *     fraction (its rays intersected with the hand plane, looked up in the hand mask) and its warmth.
 * Both stages only accept physically possible rigs ([plausible]); that rules out the planar twin, which a
 * hand held at one depth otherwise allows (a mirrored camera ~twice the hand depth out, facing the phone).
 * A tilt and a sideways offset look alike when the hand stays at one depth, so both stages also carry a weak
 * preference for small offsets and a lens near its datasheet ([OFFSET_PRIOR_CM], [LENS_PRIOR]); hand depth
 * variation (parallax) outweighs it, which is why the capture asks for the hand near and far.
 */
object ThermalCalibration {
    const val MASK_GRID = 96
    const val HAND_AREA_CM2 = 130.0          // silhouette area of an open adult hand
    const val MIN_CONTRAST_C = 4.0
    const val BLUR_PX = 0.8
    private const val PIXELS = ThermalGeometry.W * ThermalGeometry.H
    const val MAX_OFFSET_CM = 40.0
    const val MAX_EDGE_FRACTION = 0.15
    /** The capture needs the hand this much further away at its far end than at its near end (p90 / p10). */
    const val MIN_DEPTH_SPREAD = 1.5
    /** Thermal packet arrival after camera capture, measured on a recorded session (230 ms); used when the
     *  capture has too little near/far motion to measure it. A property of firmware + USB, not of the mount. */
    const val DEFAULT_LATENCY_MS = 225
    private const val MIN_LATENCY_MS = 100     // >= one 125 ms subpage integration minus slack
    private const val MAX_LATENCY_MS = 350
    private const val MIN_LAG_CORRELATION = 0.6
    const val OFFSET_PRIOR_CM = 15.0
    const val LENS_PRIOR = 0.2
    private const val PRIOR_WEIGHT = 0.01
    private val MIN_AXIS_COS = kotlin.math.cos(Math.toRadians(80.0))
    private val MAX_LOG_K = kotlin.math.ln(1.33)

    /**
     * Physically possible rig (not a mounting assumption): both cameras face the same way (optical axes
     * within 80 deg), the thermal camera is within [MAX_OFFSET_CM] of the lens and on the phone's side of the
     * hand (5 cm short of [nearCm]), and its lens within +-33% of the datasheet.
     */
    fun plausible(p: DoubleArray, nearCm: Double): Boolean {
        val r = ThermalGeometry.rotation(p[0], p[1], p[2])
        return r[8] > MIN_AXIS_COS && sqrt(p[3] * p[3] + p[4] * p[4] + p[5] * p[5]) < MAX_OFFSET_CM &&
            p[5] < nearCm - 5 && abs(p[6]) < MAX_LOG_K
    }

    /** Weak preference for small offsets and a datasheet lens, in correlation units (20 cm costs ~0.02). */
    private fun prior(p: DoubleArray): Double =
        PRIOR_WEIGHT * ((p[3] * p[3] + p[4] * p[4] + p[5] * p[5]) / (OFFSET_PRIOR_CM * OFFSET_PRIOR_CM) +
            p[6] * p[6] / (LENS_PRIOR * LENS_PRIOR))

    /** Hand depth spread of the capture (90th / 10th percentile); parallax needs it well above 1. */
    fun depthSpread(cams: List<CameraSample>): Double {
        if (cams.size < 5) return 1.0
        val d = cams.map { it.depth }.sorted()
        return d[(d.size * 0.9).toInt().coerceAtMost(d.size - 1)] / d[(d.size * 0.1).toInt()]
    }

    /** 10th percentile hand depth: the near end of where the hand was. */
    private fun nearDepth(pairs: List<Pair>): Double {
        val d = pairs.map { it.cam.depth }.sorted()
        return d[(d.size * 0.1).toInt().coerceAtMost(d.size - 1)]
    }
    private val SUB = doubleArrayOf(-0.25, -0.25, 0.25, -0.25, -0.25, 0.25, 0.25, 0.25)

    /** One analysed camera frame with a hand: mask in the model crop at 96x96, the crop box, depth, centroid. */
    class CameraSample(
        val tsNs: Long, val mask: FloatArray, val left: Int, val top: Int, val side: Int,
        val depth: Double, val point: DoubleArray, val areaPx: Double,
    )

    class ThermalSample(val arrivalNs: Long, val celsius: FloatArray)

    class Pair(val cam: CameraSample, val warm: FloatArray, val thPt: DoubleArray)

    class Result(
        val pose: ThermalPose,
        val latencyMs: Int,
        val correlation: Double,
        val stage1Px: Double,
        /** Rough 1-sigma: yaw, pitch, roll (rad), x, y, z (cm), focal scale. */
        val sigma: DoubleArray,
        val pairs: Int,
        val latencyMeasured: Boolean = true,
    )

    /**
     * Builds a sample from the model's hand mask ([size]x[size] probabilities over the crop box), or null if
     * there is no usable hand. Depth comes from the mask area: an open hand is about [HAND_AREA_CM2].
     */
    fun cameraSample(tsNs: Long, mask: FloatArray, size: Int, left: Int, top: Int, side: Int,
                     cam: CameraIntrinsics): CameraSample? {
        var above = 0
        for (p in mask) if (p > 0.5f) above++
        val areaPx = above.toDouble() / mask.size * side * side
        if (areaPx < 500) return null
        val small = downsample(mask, size, MASK_GRID)
        // A hand cut off by the box edge has a too-small area, i.e. a too-large depth: skip those.
        var edge = 0
        for (k in 0 until MASK_GRID) {
            if (small[k] > 0.5f) edge++
            if (small[(MASK_GRID - 1) * MASK_GRID + k] > 0.5f) edge++
            if (small[k * MASK_GRID] > 0.5f) edge++
            if (small[k * MASK_GRID + MASK_GRID - 1] > 0.5f) edge++
        }
        if (edge > MAX_EDGE_FRACTION * 4 * MASK_GRID) return null
        val c = blobCentroid(small, MASK_GRID, MASK_GRID, 0.5f) ?: return null
        val z = cam.f * sqrt(HAND_AREA_CM2 / areaPx)
        val px = left + (c[0] + 0.5) / MASK_GRID * side
        val py = top + (c[1] + 0.5) / MASK_GRID * side
        return CameraSample(tsNs, small, left, top, side, z,
            doubleArrayOf((px - cam.cx) / cam.f * z, (py - cam.cy) / cam.f * z, z), areaPx)
    }

    /** Area-average [size]^2 -> [grid]^2 (size must be a multiple of grid). */
    fun downsample(src: FloatArray, size: Int, grid: Int): FloatArray {
        val k = size / grid
        val out = FloatArray(grid * grid)
        for (gy in 0 until grid) for (gx in 0 until grid) {
            var s = 0f
            for (y in gy * k until gy * k + k) for (x in gx * k until gx * k + k) s += src[y * size + x]
            out[gy * grid + gx] = s / (k * k)
        }
        return out
    }

    /** 0 (background) .. 1 (fully hand) per pixel, or null if nothing in the frame is warm. */
    fun warmth(celsius: FloatArray): FloatArray? {
        val sorted = celsius.sortedArray()
        val lo = percentile(sorted, 20.0)
        val hi = percentile(sorted, 98.0)
        if (hi - lo < MIN_CONTRAST_C) return null
        return FloatArray(celsius.size) { ((celsius[it] - lo) / (hi - lo)).coerceIn(0.0, 1.0).toFloat() }
    }

    /** numpy-style linear-interpolated percentile of a sorted array. */
    private fun percentile(sorted: FloatArray, q: Double): Double {
        val pos = q / 100 * (sorted.size - 1)
        val i = pos.toInt()
        val f = pos - i
        return if (i + 1 < sorted.size) sorted[i] * (1 - f) + sorted[i + 1] * f else sorted[i].toDouble()
    }

    /** Weighted centroid (x, y) of the largest 8-connected region above [threshold]. */
    fun blobCentroid(img: FloatArray, w: Int, h: Int, threshold: Float): DoubleArray? {
        val labels = IntArray(w * h)
        val queue = IntArray(w * h)
        var best = 0
        var bestSize = 0
        var next = 0
        for (start in 0 until w * h) {
            if (img[start] <= threshold || labels[start] != 0) continue
            next++
            var head = 0
            var tail = 0
            queue[tail++] = start
            labels[start] = next
            while (head < tail) {
                val i = queue[head++]
                val x = i % w
                val y = i / w
                for (dy in -1..1) for (dx in -1..1) {
                    val nx = x + dx
                    val ny = y + dy
                    if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue
                    val j = ny * w + nx
                    if (labels[j] == 0 && img[j] > threshold) {
                        labels[j] = next
                        queue[tail++] = j
                    }
                }
            }
            if (tail > bestSize) {
                bestSize = tail
                best = next
            }
        }
        if (best == 0) return null
        var sw = 0.0
        var sx = 0.0
        var sy = 0.0
        for (i in 0 until w * h) if (labels[i] == best) {
            sw += img[i]; sx += img[i] * (i % w); sy += img[i] * (i / w)
        }
        return doubleArrayOf(sx / sw, sy / sw)
    }

    /** One pair per thermal frame: the camera frame captured [latencyMs] before the packet arrived. */
    fun makePairs(cams: List<CameraSample>, thermals: List<ThermalSample>, latencyMs: Int): List<Pair> {
        if (cams.isEmpty()) return emptyList()
        val sortedCams = cams.sortedBy { it.tsNs }
        val ts = LongArray(sortedCams.size) { sortedCams[it].tsNs }
        val out = ArrayList<Pair>()
        for (t in thermals) {
            val target = t.arrivalNs - latencyMs * 1_000_000L
            var i = java.util.Arrays.binarySearch(ts, target).let { if (it < 0) -it - 1 else it }
            i = i.coerceIn(1, max(1, ts.size - 1))
            if (ts.size == 1) i = 0 else if (abs(ts[i - 1] - target) <= abs(ts[i] - target)) i -= 1
            if (abs(ts[i] - target) > 40_000_000L) continue
            val warm = warmth(t.celsius) ?: continue
            val c = blobCentroid(warm, ThermalGeometry.W, ThermalGeometry.H, 0.5f) ?: continue
            out.add(Pair(sortedCams[i], warm, c))
        }
        return out
    }

    // ------------------------------------------------------------------------------------------ stage 0

    /** Hand size (camera) vs warm area (thermal) cross-correlation peak, or null if too weak to trust. */
    fun estimateLatency(cams: List<CameraSample>, thermals: List<ThermalSample>): kotlin.Pair<Int, Double>? {
        if (cams.size < 10 || thermals.size < 10) return null
        val c = cams.sortedBy { it.tsNs }
        val t = thermals.sortedBy { it.arrivalNs }
        val cx = c.map { it.tsNs.toDouble() }.toDoubleArray()
        val cy = c.map { it.areaPx }.toDoubleArray()
        val tx = t.map { it.arrivalNs.toDouble() }.toDoubleArray()
        val ty = t.map { s ->
            val sorted = s.celsius.sortedArray()
            val lo = percentile(sorted, 20.0)
            s.celsius.count { it - lo > MIN_CONTRAST_C }.toDouble()
        }.toDoubleArray()
        val grid = generateSequence(cx.first()) { it + 20e6 }.takeWhile { it <= cx.last() }.toList()
        if (grid.size < 20) return null
        val a = grid.map { interp(it, cx, cy) }
        var best = -1.0
        var bestLag = 0
        for (lag in MIN_LATENCY_MS..MAX_LATENCY_MS step 10) {
            val b = grid.map { interp(it + lag * 1e6, tx, ty) }
            val c0 = pearson(a, b)
            if (c0 > best) {
                best = c0; bestLag = lag
            }
        }
        return if (best >= MIN_LAG_CORRELATION) kotlin.Pair(bestLag, best) else null
    }

    private fun interp(x: Double, xs: DoubleArray, ys: DoubleArray): Double {
        if (x <= xs[0]) return ys[0]
        if (x >= xs[xs.size - 1]) return ys[ys.size - 1]
        var i = java.util.Arrays.binarySearch(xs, x)
        if (i >= 0) return ys[i]
        i = -i - 1
        val f = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
        return ys[i - 1] * (1 - f) + ys[i] * f
    }

    private fun pearson(a: List<Double>, b: List<Double>): Double {
        val ma = a.average()
        val mb = b.average()
        var sab = 0.0; var saa = 0.0; var sbb = 0.0
        for (i in a.indices) {
            val da = a[i] - ma
            val db = b[i] - mb
            sab += da * db; saa += da * da; sbb += db * db
        }
        return if (saa > 0 && sbb > 0) sab / sqrt(saa * sbb) else 0.0
    }

    // ------------------------------------------------------------------------------------------ stage 1

    private const val HUBER = 1.5

    /** Robust cost and its Gauss-Newton pieces for the centroid reprojection; [x] = 6 pose params. */
    private fun reprojection(x: DoubleArray, mirror: Boolean, pairs: List<Pair>, residual: DoubleArray) {
        val pose = ThermalPose(x[0], x[1], x[2], x[3], x[4], x[5], 0.0, mirror)
        val uv = DoubleArray(2)
        for ((i, p) in pairs.withIndex()) {
            pose.project(p.cam.point[0], p.cam.point[1], p.cam.point[2], uv)
            residual[2 * i] = uv[0] - p.thPt[0]
            residual[2 * i + 1] = uv[1] - p.thPt[1]
        }
        // Weak tie-breaker towards small offsets: 1 px per OFFSET_PRIOR_CM.
        for (k in 0..2) residual[2 * pairs.size + k] = x[3 + k] / OFFSET_PRIOR_CM
    }

    private fun huberCost(r: DoubleArray): Double {
        var c = 0.0
        for (e in r) {
            val a = abs(e)
            c += if (a <= HUBER) 0.5 * e * e else HUBER * (a - 0.5 * HUBER)
        }
        return c
    }

    /** Huber-weighted Levenberg-Marquardt with a forward-difference Jacobian. Returns (params, cost). */
    private fun levenbergMarquardt(x0: DoubleArray, mirror: Boolean, pairs: List<Pair>, maxIter: Int): kotlin.Pair<DoubleArray, Double> {
        val n = 6
        val m = pairs.size * 2 + 3
        var x = x0.copyOf()
        val r = DoubleArray(m)
        val rp = DoubleArray(m)
        val jac = Array(n) { DoubleArray(m) }
        reprojection(x, mirror, pairs, r)
        var cost = huberCost(r)
        var lambda = 1e-3
        repeat(maxIter) {
            for (k in 0 until n) {
                val h = 1e-6 * max(1.0, abs(x[k]))
                val xp = x.copyOf().also { it[k] += h }
                reprojection(xp, mirror, pairs, rp)
                for (j in 0 until m) jac[k][j] = (rp[j] - r[j]) / h
            }
            val a = Array(n) { DoubleArray(n) }
            val g = DoubleArray(n)
            for (j in 0 until m) {
                val e = abs(r[j])
                val w = if (e <= HUBER) 1.0 else HUBER / e
                for (k in 0 until n) {
                    g[k] += w * jac[k][j] * r[j]
                    for (l in k until n) a[k][l] += w * jac[k][j] * jac[l][j]
                }
            }
            for (k in 0 until n) for (l in 0 until k) a[k][l] = a[l][k]
            var improved = false
            while (lambda < 1e10) {
                val aa = Array(n) { k -> DoubleArray(n) { l -> a[k][l] + if (k == l) lambda * max(a[k][k], 1e-9) else 0.0 } }
                val step = solve(aa, DoubleArray(n) { -g[it] }) ?: break
                val xn = DoubleArray(n) { x[it] + step[it] }
                reprojection(xn, mirror, pairs, rp)
                val cn = huberCost(rp)
                if (cn < cost) {
                    val rel = (cost - cn) / max(cost, 1e-12)
                    x = xn
                    System.arraycopy(rp, 0, r, 0, m)
                    cost = cn
                    lambda = max(lambda / 3, 1e-9)
                    improved = rel > 1e-9
                    break
                }
                lambda *= 4
            }
            if (!improved) return kotlin.Pair(x, cost)
        }
        return kotlin.Pair(x, cost)
    }

    /** Gaussian elimination with partial pivoting; null if singular. */
    private fun solve(a: Array<DoubleArray>, b: DoubleArray): DoubleArray? {
        val n = b.size
        for (c in 0 until n) {
            var p = c
            for (r in c + 1 until n) if (abs(a[r][c]) > abs(a[p][c])) p = r
            if (abs(a[p][c]) < 1e-15) return null
            a[c] = a[p].also { a[p] = a[c] }
            b[c] = b[p].also { b[p] = b[c] }
            for (r in c + 1 until n) {
                val f = a[r][c] / a[c][c]
                for (k in c until n) a[r][k] -= f * a[c][k]
                b[r] -= f * b[c]
            }
        }
        val x = DoubleArray(n)
        for (r in n - 1 downTo 0) {
            var s = b[r]
            for (k in r + 1 until n) s -= a[r][k] * x[k]
            x[r] = s / a[r][r]
        }
        return x
    }

    private fun medianReprojection(x: DoubleArray, mirror: Boolean, pairs: List<Pair>): Double {
        val r = DoubleArray(pairs.size * 2 + 3)
        reprojection(x, mirror, pairs, r)
        return DoubleArray(pairs.size) { kotlin.math.hypot(r[2 * it], r[2 * it + 1]) }.sorted()[pairs.size / 2]
    }

    /** Multi-start over all rolls x yaw/pitch +-60 deg x mirror. Returns (params[7], mirror, median px). */
    fun poseFromPoints(pairs: List<Pair>): Triple<DoubleArray, Boolean, Double> {
        val near = nearDepth(pairs)
        var best = Double.POSITIVE_INFINITY
        var bestX = DoubleArray(6)
        var bestMirror = false
        for (mirror in listOf(false, true)) for (rollDeg in 0 until 360 step 30)
            for (yawDeg in -60..60 step 30) for (pitchDeg in -60..60 step 30) {
                val x0 = doubleArrayOf(Math.toRadians(yawDeg.toDouble()), Math.toRadians(pitchDeg.toDouble()),
                    Math.toRadians(rollDeg.toDouble()), 0.0, 0.0, 0.0)
                val (x, cost) = levenbergMarquardt(x0, mirror, pairs, 60)
                if (cost < best && plausible(x + 0.0, near)) {
                    best = cost; bestX = x; bestMirror = mirror
                }
            }
        bestX[2] = wrapAngle(bestX[2])
        return Triple(bestX + 0.0, bestMirror, medianReprojection(bestX, bestMirror, pairs))
    }

    /** Same rotation, reported with |pitch| <= 90 deg: (yaw, pitch, roll) == (yaw+180, 180-pitch, roll+180). */
    fun canonicalize(p: DoubleArray) {
        p[1] = wrapAngle(p[1])
        if (abs(p[1]) > Math.PI / 2) {
            p[0] += Math.PI
            p[1] = Math.PI - p[1]
            p[2] += Math.PI
        }
        for (i in 0..2) p[i] = wrapAngle(p[i])
    }

    private fun wrapAngle(a: Double): Double {
        var r = (a + Math.PI) % (2 * Math.PI)
        if (r < 0) r += 2 * Math.PI
        return r - Math.PI
    }

    // ------------------------------------------------------------------------------------------ stage 2

    /** Sub-pixel rays (768 x 4 x 3) of [pose] in camera coordinates, pixel-major. */
    fun rays(pose: ThermalPose): DoubleArray {
        val rays = DoubleArray(PIXELS * 4 * 3)
        val d = DoubleArray(3)
        for (v in 0 until ThermalGeometry.H) for (u in 0 until ThermalGeometry.W) for (s in 0 until 4) {
            pose.ray(u + SUB[2 * s], v + SUB[2 * s + 1], d)
            val o = ((v * ThermalGeometry.W + u) * 4 + s) * 3
            rays[o] = d[0]; rays[o + 1] = d[1]; rays[o + 2] = d[2]
        }
        return rays
    }

    /**
     * Predicted hand fraction per thermal pixel (blurred by the optics, into [frac]) for one camera sample,
     * and whether all of the pixel's rays land inside the crop box (into [valid]).
     */
    fun predict(rays: DoubleArray, p: DoubleArray, c: CameraSample, cam: CameraIntrinsics,
                frac: FloatArray, valid: BooleanArray) {
        val raw = FloatArray(PIXELS)
        val z = c.depth
        val m = c.mask
        for (i in 0 until PIXELS) {
            var acc = 0f
            var ok = true
            for (s in 0 until 4) {
                val o = (i * 4 + s) * 3
                val scale = (z - p[5]) / rays[o + 2]
                val px = cam.f * (p[3] + scale * rays[o]) / z + cam.cx
                val py = cam.f * (p[4] + scale * rays[o + 1]) / z + cam.cy
                var gx = (px - c.left) / c.side * MASK_GRID - 0.5
                var gy = (py - c.top) / c.side * MASK_GRID - 0.5
                if (!(scale > 0 && gx >= 0 && gx <= MASK_GRID - 1 && gy >= 0 && gy <= MASK_GRID - 1)) ok = false
                if (gx.isNaN()) gx = 0.0
                if (gy.isNaN()) gy = 0.0
                gx = gx.coerceIn(0.0, MASK_GRID - 1.001)
                gy = gy.coerceIn(0.0, MASK_GRID - 1.001)
                val x0 = gx.toInt()
                val y0 = gy.toInt()
                val fx = (gx - x0).toFloat()
                val fy = (gy - y0).toFloat()
                val b = y0 * MASK_GRID + x0
                acc += m[b] * (1 - fx) * (1 - fy) + m[b + 1] * fx * (1 - fy) +
                    m[b + MASK_GRID] * (1 - fx) * fy + m[b + MASK_GRID + 1] * fx * fy
            }
            raw[i] = acc / 4
            valid[i] = ok
        }
        gaussianBlur(raw, frac)
    }

    /** Pearson correlation between predicted hand fraction and warmth over pixels whose rays hit the crop. */
    fun correlation(p: DoubleArray, mirror: Boolean, pairs: List<Pair>, cam: CameraIntrinsics): Double {
        val rays = rays(ThermalPose.of(p, mirror))
        val frac = FloatArray(PIXELS)
        val valid = BooleanArray(PIXELS)
        var n = 0.0; var sx = 0.0; var sy = 0.0; var sxx = 0.0; var syy = 0.0; var sxy = 0.0
        for (pair in pairs) {
            predict(rays, p, pair.cam, cam, frac, valid)
            for (i in 0 until PIXELS) if (valid[i]) {
                val a = frac[i].toDouble()
                val b = pair.warm[i].toDouble()
                n++; sx += a; sy += b; sxx += a * a; syy += b * b; sxy += a * b
            }
        }
        if (n < 50) return -1.0
        val cov = sxy - sx * sy / n
        val va = sxx - sx * sx / n
        val vb = syy - sy * sy / n
        if (va <= 0 || vb <= 0) return -1.0
        return cov / sqrt(va * vb)
    }

    private val KERNEL: FloatArray = run {
        val radius = (4 * BLUR_PX + 0.5).toInt()                  // scipy gaussian_filter, truncate = 4
        val k = FloatArray(2 * radius + 1) { exp(-0.5 * ((it - radius) / BLUR_PX).let { x -> x * x }).toFloat() }
        val s = k.sum()
        FloatArray(k.size) { k[it] / s }
    }

    /** Separable Gaussian over the 32x24 grid, scipy "reflect" borders (d c b a | a b c d). */
    private fun gaussianBlur(src: FloatArray, dst: FloatArray) {
        val w = ThermalGeometry.W
        val h = ThermalGeometry.H
        val r = KERNEL.size / 2
        val tmp = FloatArray(src.size)
        fun reflect(i: Int, n: Int): Int {
            var j = i
            while (j < 0 || j >= n) j = if (j < 0) -j - 1 else 2 * n - j - 1
            return j
        }
        for (y in 0 until h) for (x in 0 until w) {
            var s = 0f
            for (k in -r..r) s += KERNEL[k + r] * src[y * w + reflect(x + k, w)]
            tmp[y * w + x] = s
        }
        for (y in 0 until h) for (x in 0 until w) {
            var s = 0f
            for (k in -r..r) s += KERNEL[k + r] * tmp[reflect(y + k, h) * w + x]
            dst[y * w + x] = s
        }
    }

    /** Adaptive Nelder-Mead (Gao & Han 2012 coefficients, as scipy's adaptive=True) minimising [f]. */
    fun nelderMead(f: (DoubleArray) -> Double, x0: DoubleArray, step: DoubleArray, maxIter: Int,
                   xatol: Double = 1e-4, fatol: Double = 1e-5): DoubleArray {
        val n = x0.size
        val rho = 1.0
        val chi = 1 + 2.0 / n
        val psi = 0.75 - 1 / (2.0 * n)
        val sigma = 1 - 1.0 / n
        val sim = Array(n + 1) { i -> x0.copyOf().also { if (i > 0) it[i - 1] += step[i - 1] } }
        val fs = DoubleArray(n + 1) { f(sim[it]) }
        repeat(maxIter) {
            val order = (0..n).sortedBy { fs[it] }
            val s2 = Array(n + 1) { sim[order[it]] }
            val f2 = DoubleArray(n + 1) { fs[order[it]] }
            for (i in 0..n) { sim[i] = s2[i]; fs[i] = f2[i] }
            var maxDx = 0.0
            var maxDf = 0.0
            for (i in 1..n) {
                for (k in 0 until n) maxDx = max(maxDx, abs(sim[i][k] - sim[0][k]))
                maxDf = max(maxDf, abs(fs[i] - fs[0]))
            }
            if (maxDx <= xatol && maxDf <= fatol) return sim[0]
            val xbar = DoubleArray(n) { k -> (0 until n).sumOf { sim[it][k] } / n }
            fun along(t: Double) = DoubleArray(n) { xbar[it] + t * (sim[n][it] - xbar[it]) }
            val xr = along(-rho)
            val fr = f(xr)
            var shrink = false
            if (fr < fs[0]) {
                val xe = along(-rho * chi)
                val fe = f(xe)
                if (fe < fr) { sim[n] = xe; fs[n] = fe } else { sim[n] = xr; fs[n] = fr }
            } else if (fr < fs[n - 1]) {
                sim[n] = xr; fs[n] = fr
            } else if (fr < fs[n]) {
                val xc = along(-psi * rho)
                val fc = f(xc)
                if (fc <= fr) { sim[n] = xc; fs[n] = fc } else shrink = true
            } else {
                val xcc = along(psi)
                val fcc = f(xcc)
                if (fcc < fs[n]) { sim[n] = xcc; fs[n] = fcc } else shrink = true
            }
            if (shrink) for (i in 1..n) {
                sim[i] = DoubleArray(n) { sim[0][it] + sigma * (sim[i][it] - sim[0][it]) }
                fs[i] = f(sim[i])
            }
        }
        return sim[(0..n).minByOrNull { fs[it] }!!]
    }

    private fun sensitivity(p: DoubleArray, mirror: Boolean, pairs: List<Pair>, cam: CameraIntrinsics): DoubleArray {
        val steps = doubleArrayOf(Math.toRadians(0.5), Math.toRadians(0.5), Math.toRadians(0.5), 0.5, 0.5, 0.5, 0.02)
        val c0 = correlation(p, mirror, pairs, cam)
        return DoubleArray(7) { i ->
            val h = steps[i]
            val plus = p.copyOf().also { it[i] += h }
            val minus = p.copyOf().also { it[i] -= h }
            val curv = (2 * c0 - correlation(plus, mirror, pairs, cam) - correlation(minus, mirror, pairs, cam)) / (h * h)
            if (curv > 0) sqrt(0.004 / curv) else Double.POSITIVE_INFINITY
        }
    }

    private fun <T> List<T>.every(target: Int): List<T> {
        val k = max(1, size / max(1, target))
        return filterIndexed { i, _ -> i % k == 0 }
    }

    /**
     * Full calibration from raw samples. [progress] gets 0..1 while solving. Returns null when there are too
     * few frames with the hand seen by both cameras.
     */
    fun solve(cams: List<CameraSample>, thermals: List<ThermalSample>, cam: CameraIntrinsics,
              maxIter: Int = 800, progress: (Double) -> Unit = {}): Result? {
        // Stage 0: latency from geometry-free signals (see the class comment for why not from the fit).
        val measured = estimateLatency(cams, thermals)
        val latency = measured?.first ?: DEFAULT_LATENCY_MS
        val pairs = makePairs(cams, thermals, latency)
        if (pairs.size < MIN_PAIRS) return null
        // Stage 1: pose from points, multi-start.
        val (x1, mirror, stage1Err) = poseFromPoints(pairs.every(120))
        progress(0.4)
        val fitSet = pairs.every(200)
        val near = nearDepth(pairs)
        val p1 = doubleArrayOf(x1[0], x1[1], wrapAngle(x1[2]), x1[3], x1[4], x1[5], 0.0)

        // Stage 2: maximise silhouette correlation.
        var evals = 0
        val step = doubleArrayOf(Math.toRadians(3.0), Math.toRadians(3.0), Math.toRadians(3.0), 2.0, 2.0, 2.0, 0.1)
        val p = nelderMead({ q ->
            evals++
            if (evals % 25 == 0) progress(min(0.95, 0.4 + 0.55 * evals / (maxIter * 1.5)))
            if (plausible(q, near)) -correlation(q, mirror, fitSet, cam) + prior(q) else 1.0
        }, p1, step, maxIter)
        val sigma = sensitivity(p, mirror, fitSet, cam)
        progress(1.0)
        canonicalize(p)
        return Result(ThermalPose.of(p, mirror), latency, correlation(p, mirror, pairs, cam), stage1Err, sigma, pairs.size,
            latencyMeasured = measured != null)
    }

    const val MIN_PAIRS = 12

    /** Saves a capture (for replaying it offline: ThermalCalibrationTest picks up ML/data/calibrations/). */
    fun writeCapture(file: java.io.File, cams: List<CameraSample>, thermals: List<ThermalSample>, cam: CameraIntrinsics) {
        java.io.DataOutputStream(java.io.BufferedOutputStream(java.io.FileOutputStream(file))).use { o ->
            o.writeInt(CAPTURE_VERSION)
            o.writeDouble(cam.f); o.writeDouble(cam.cx); o.writeDouble(cam.cy)
            o.writeInt(cams.size)
            for (c in cams) {
                o.writeLong(c.tsNs); o.writeInt(c.left); o.writeInt(c.top); o.writeInt(c.side)
                o.writeDouble(c.depth); for (v in c.point) o.writeDouble(v); o.writeDouble(c.areaPx)
                for (v in c.mask) o.writeByte((v.coerceIn(0f, 1f) * 255).toInt())
            }
            o.writeInt(thermals.size)
            for (t in thermals) {
                o.writeLong(t.arrivalNs)
                for (v in t.celsius) o.writeFloat(v)
            }
        }
    }

    fun readCapture(file: java.io.File): Triple<List<CameraSample>, List<ThermalSample>, CameraIntrinsics> {
        java.io.DataInputStream(java.io.BufferedInputStream(java.io.FileInputStream(file))).use { i ->
            require(i.readInt() == CAPTURE_VERSION)
            val cam = CameraIntrinsics(i.readDouble(), i.readDouble(), i.readDouble())
            val cams = List(i.readInt()) {
                val ts = i.readLong(); val l = i.readInt(); val t = i.readInt(); val sd = i.readInt()
                val depth = i.readDouble()
                val point = DoubleArray(3) { i.readDouble() }
                val area = i.readDouble()
                val mask = FloatArray(MASK_GRID * MASK_GRID) { (i.readUnsignedByte()) / 255f }
                CameraSample(ts, mask, l, t, sd, depth, point, area)
            }
            val thermals = List(i.readInt()) { ThermalSample(i.readLong(), FloatArray(PIXELS) { i.readFloat() }) }
            return Triple(cams, thermals, cam)
        }
    }

    private const val CAPTURE_VERSION = 1
}
