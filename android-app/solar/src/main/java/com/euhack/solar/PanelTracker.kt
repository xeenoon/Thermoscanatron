package com.euhack.solar

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.floor

/**
 * Panel profile: the cell lattice and its physical pitch. Same as ML/src/segkit/panel/spec.py.
 * Panel coordinates: u across the columns (0..cols), v down the rows (0..rows), in cells.
 */
data class PanelSpec(
    val cols: Int = 4,
    val rows: Int = 9,
    val cellWcm: Double = 15.6,
    val cellHcm: Double = 10.8,
)

/**
 * Which cell is which, from PanelNet's per-pixel outputs. Port of ML/src/segkit/panel/track.py, working in
 * model-crop pixels (size x size); H maps panel (u, v) to crop pixels.
 *
 *  acquire  enough of the outline in view (a corner or two crossing edges): unwrap the within-cell phase
 *           across the visible panel, fit H (known up to a whole cell in u and two rows in v), then choose the
 *           integer offset whose cell area best matches the predicted panel mask.
 *  follow   predict H from the last one (and the gyro's rotation since, when given), unwrap each panel pixel's
 *           phase against that prediction, RANSAC-fit H. The model pins the position inside the cell every
 *           frame; the prediction only has to be within half a cell (two rows for v).
 *  correct  off-by-one candidates are scored against the predicted mask, so a visible edge fixes any slip.
 *  lost     after [LOST_AFTER] frames without a good fit.
 */
class PanelTracker(val spec: PanelSpec, val size: Int) {
    enum class State { LOST, ACQUIRED, TRACKING, RELOCKED, COASTING }

    var h: DoubleArray? = null
        private set
    var state = State.LOST
        private set
    private var misses = 0

    private val g = size / STRIDE
    private val mask = BooleanArray(g * g)
    private val pu = DoubleArray(g * g)
    private val pv = DoubleArray(g * g)
    private val xy = DoubleArray(2 * g * g)
    private val uv = DoubleArray(2 * g * g)
    private var n = 0
    private val tmp = DoubleArray(2)
    /** Debug: panel samples this frame, and each motion prior's inlier fraction (-1: no fit). */
    var lastSamples = 0
        private set
    var lastFits = DoubleArray(0)
        private set

    fun reset() {
        h = null
        state = State.LOST
        misses = 0
    }

    /**
     * One frame. [dense] is the model's [6, size, size] output (mask prob, line prob, unit phase pairs),
     * [predictions] candidate motion models for this frame (each maps last frame's crop to this one's), the
     * tracker keeps whichever explains the phase best.
     */
    fun step(dense: FloatArray, present: Float, predictions: List<DoubleArray>, denseSize: Int = size,
             coastOnly: Boolean = false): State {
        sample(dense, denseSize)
        if (coastOnly && h != null) {
            // Motion blur: the model's view is smeared, follow the motion prior alone.
            h = Mat3.mul(predictions.first(), h!!)
            state = State.COASTING
            return state
        }
        lastSamples = n
        val prev = h
        if (prev != null) {
            var best: HomographyFit.Result? = null
            lastFits = DoubleArray(predictions.size) { -1.0 }
            if (present > 0.5f) {
                // Cheap score for every motion prior (how much of the phase field it already explains), then the
                // full RANSAC fit only for the best one.
                var bestJ = 0
                for ((j, m) in predictions.withIndex()) {
                    lastFits[j] = agreement(Mat3.mul(m, prev))
                    if (lastFits[j] > lastFits[bestJ]) bestJ = j
                }
                best = fit(Mat3.mul(predictions[bestJ], prev))
            }
            // Coast on the motion prior that last explained the frame best (the gyro, once it has).
            val pred = Mat3.mul(predictions[bestPrior(best, predictions.size)], prev)
            val relocked = if (best == null || best.inlierFraction < MIN_INLIER_FRAC) {
                if (present > 0.5f) relock(pred) else null
            } else null
            if (best != null && best.inlierFraction >= MIN_INLIER_FRAC) {
                h = correct(best.h)
                misses = 0
                state = State.TRACKING
            } else if (relocked != null) {
                h = correct(relocked)
                misses = 0
                state = State.RELOCKED
            } else {
                h = pred
                misses++
                state = State.COASTING
                val onPanel = present > 0.5f && n > ON_PANEL_FRACTION * g * g
                if (misses > (if (onPanel) LOST_AFTER_ON_PANEL else LOST_AFTER)) {
                    h = null
                    state = State.LOST
                }
            }
        }
        if (h == null && present > 0.5f) {
            acquire()?.let {
                h = it
                misses = 0
                state = State.ACQUIRED
            }
        }
        return state
    }

    private var lastGoodPrior = 0

    /** Index of the motion prior to coast on: this frame's best fit if any, else the last one that won. */
    private fun bestPrior(best: HomographyFit.Result?, count: Int): Int {
        if (best != null) {
            var j = 0
            for (k in lastFits.indices) if (lastFits[k] > lastFits[j]) j = k
            lastGoodPrior = j
        }
        return if (lastGoodPrior < count) lastGoodPrior else 0
    }

    /**
     * The prediction has gone stale (e.g. moving in fast): rebuild H from this frame's phase field alone and take
     * the integer cell offset that puts the crop centre where the prediction had it. Moving in scales about the
     * centre, so the centre cell is the part of a stale prediction that is still right.
     */
    private fun relock(pred: DoubleArray): DoubleArray? {
        val r = unwrappedFit() ?: return null
        val c = size / 2.0
        Mat3.apply(Mat3.inv(pred), c, c, tmp)
        val wu = tmp[0]; val wv = tmp[1]
        Mat3.apply(Mat3.inv(r), c, c, tmp)
        val du = Math.round(wu - tmp[0]).toDouble()
        val dv = 2.0 * Math.round((wv - tmp[1]) / 2)
        return Mat3.mul(r, Mat3.translate(-du, -dv))
    }

    /** (row, col) of the cell under crop pixel (x, y), or null. */
    fun cellAt(x: Double, y: Double): IntArray? {
        val hh = h ?: return null
        val w = Mat3.apply(Mat3.inv(hh), x, y, tmp)
        if (w <= 0) return null
        val u = tmp[0]; val v = tmp[1]
        if (u < 0 || u >= spec.cols || v < 0 || v >= spec.rows) return null
        return intArrayOf(v.toInt(), u.toInt())
    }

    // ---------------------------------------------------------------------------------------------------

    /** Seed (or clear) the panel homography, e.g. from the other model's tracker. */
    fun seed(hh: DoubleArray?) {
        h = hh?.copyOf()
        misses = 0
        state = if (hh == null) State.LOST else State.TRACKING
    }

    /** [dense] may come from a smaller model ([denseSize] < [size]): samples are taken at the matching spot. */
    private fun sample(dense: FloatArray, denseSize: Int) {
        val plane = denseSize * denseSize
        val k0 = denseSize.toDouble() / size
        n = 0
        for (gy in 0 until g) for (gx in 0 until g) {
            val px = ((gx * STRIDE + STRIDE / 2) * k0).toInt()
            val py = ((gy * STRIDE + STRIDE / 2) * k0).toInt()
            val i = py * denseSize + px
            val k = gy * g + gx
            mask[k] = dense[i] > 0.5f
            pu[k] = wrap(atan2(dense[2 * plane + i].toDouble(), dense[3 * plane + i].toDouble()) / (2 * PI), 1.0)
            pv[k] = wrap(atan2(dense[4 * plane + i].toDouble(), dense[5 * plane + i].toDouble()) / PI, 2.0)
            if (mask[k]) n++
        }
    }

    /** Share of (every other) panel sample whose phase is within [AGREE_CELLS] of what [pred] says it should be. */
    private fun agreement(pred: DoubleArray): Double {
        val inv = Mat3.inv(pred)
        var hit = 0
        var total = 0
        var k = 0
        while (k < g * g) {
            if (mask[k]) {
                total++
                val x = (k % g) * STRIDE + STRIDE / 2 + 0.5
                val y = (k / g) * STRIDE + STRIDE / 2 + 0.5
                if (Mat3.apply(inv, x, y, tmp) > 0) {
                    val du = wrap(tmp[0] - pu[k] + 0.5, 1.0) - 0.5
                    val dv = wrap(tmp[1] - pv[k] + 1.0, 2.0) - 1.0
                    if (abs(du) < AGREE_CELLS && abs(dv) < AGREE_CELLS) hit++
                }
            }
            k += 2
        }
        return if (total > 0) hit.toDouble() / total else 0.0
    }

    /** Unwrap every panel sample against [pred] and RANSAC-fit (u, v) -> crop pixels. */
    private fun fit(pred: DoubleArray): HomographyFit.Result? {
        if (n < MIN_FIT_POINTS) return null
        val inv = Mat3.inv(pred)
        var m = 0
        for (k in 0 until g * g step 2) {
            if (!mask[k]) continue
            val x = (k % g) * STRIDE + STRIDE / 2 + 0.5
            val y = (k / g) * STRIDE + STRIDE / 2 + 0.5
            if (Mat3.apply(inv, x, y, tmp) <= 0) continue
            uv[2 * m] = pu[k] + Math.round(tmp[0] - pu[k])
            uv[2 * m + 1] = pv[k] + 2.0 * Math.round((tmp[1] - pv[k]) / 2)
            xy[2 * m] = x
            xy[2 * m + 1] = y
            m++
        }
        return HomographyFit.ransac(uv, xy, m, ransacPx(m))
    }

    /** Inlier threshold: [RANSAC_CELLS] of the local cell size (affine least-squares uv -> xy), >= [RANSAC_PX]. */
    private fun ransacPx(m: Int): Double {
        if (m < 3) return RANSAC_PX
        // Normal equations for x = a u + b v + c and y = d u + e v + f.
        val ata = DoubleArray(9)
        val atx = DoubleArray(3)
        val aty = DoubleArray(3)
        for (i in 0 until m) {
            val r = doubleArrayOf(uv[2 * i], uv[2 * i + 1], 1.0)
            for (a in 0 until 3) {
                for (b in 0 until 3) ata[a * 3 + b] += r[a] * r[b]
                atx[a] += r[a] * xy[2 * i]
                aty[a] += r[a] * xy[2 * i + 1]
            }
        }
        val inv = Mat3.inv(ata)
        if (!inv.all { it.isFinite() }) return RANSAC_PX
        fun solve(b: DoubleArray) = DoubleArray(3) { r -> inv[r * 3] * b[0] + inv[r * 3 + 1] * b[1] + inv[r * 3 + 2] * b[2] }
        val px = solve(atx)
        val py = solve(aty)
        val cell = Math.sqrt(abs(px[0] * py[1] - px[1] * py[0]))
        return maxOf(RANSAC_PX, RANSAC_CELLS * cell)
    }

    /**
     * Renumber the cells if a visible panel edge or corner says so: the whole-cell offset (u + du, v + dv, dv even)
     * whose cell area matches the predicted mask clearly better than the current numbering wins. With no edge in
     * view all offsets tie and the numbering carried over from earlier frames stays.
     */
    private fun correct(hh: DoubleArray): DoubleArray {
        val s = offsetIous(hh)
        val current = s[offsetIndex(0, 0)]
        var best = 0
        for (k in s.indices) if (s[k] > s[best]) best = k
        if (s[best] <= current + SHIFT_MARGIN) return hh
        return Mat3.mul(hh, Mat3.translate(-offsetDu(best).toDouble(), -offsetDv(best).toDouble()))
    }

    private val nDv = (spec.rows + 2) / 2 * 2 + 1   // dv = -rows-1 .. rows+1 step 2
    private fun offsetIndex(du: Int, dv: Int) = (du + spec.cols) * nDv + (dv + spec.rows + 1) / 2
    private fun offsetDu(k: Int) = k / nDv - spec.cols
    private fun offsetDv(k: Int) = (k % nDv) * 2 - spec.rows - 1

    /** IoU of the cell area with the predicted mask for every whole-cell renumbering of [hh], on the sample grid:
     *  shifting the numbering just adds integers to each sample's (u, v), so (u, v) is computed once. */
    private fun offsetIous(hh: DoubleArray): DoubleArray {
        val inv = Mat3.inv(hh)
        val total = (2 * spec.cols + 1) * nDv
        val inter = IntArray(total)
        val inside = IntArray(total)
        var maskCount = 0
        for (k in 0 until g * g step 2) {
            if ((k / g) % 2 == 1) continue      // every other row and column: plenty for an IoU
            if (mask[k]) maskCount++
            val x = (k % g) * STRIDE + STRIDE / 2 + 0.5
            val y = (k / g) * STRIDE + STRIDE / 2 + 0.5
            if (Mat3.apply(inv, x, y, tmp) <= 0) continue
            val u = tmp[0]; val v = tmp[1]
            for (du in -spec.cols..spec.cols) {
                val uu = u + du
                if (uu < 0 || uu > spec.cols) continue
                var dv = -spec.rows - 1
                while (dv <= spec.rows + 1) {
                    val vv = v + dv
                    if (vv >= 0 && vv <= spec.rows) {
                        val o = offsetIndex(du, dv)
                        inside[o]++
                        if (mask[k]) inter[o]++
                    }
                    dv += 2
                }
            }
        }
        // union = inside + mask - inter
        return DoubleArray(total) { o ->
            val union = inside[o] + maskCount - inter[o]
            if (union > 0) inter[o].toDouble() / union else 0.0
        }
    }

    /** H from the phase field alone, spatially unwrapped: right up to a whole cell in u and two rows in v. */
    private fun unwrappedFit(): DoubleArray? {
        if (n < MIN_FIT_POINTS) return null
        // Seed: the panel sample furthest from the mask's edge (multi-source BFS from everything off the panel).
        var seedAll = false
        val dist = IntArray(g * g) { -1 }
        val queue = IntArray(g * g)
        var head = 0
        var tail = 0
        for (k in 0 until g * g) if (!mask[k]) { dist[k] = 0; queue[tail++] = k }
        if (tail == 0) {   // panel fills the whole crop: seed in the middle
            seedAll = true
        }
        while (head < tail) {
            val k = queue[head++]
            for (nb in neighbours(k)) if (nb >= 0 && dist[nb] < 0) { dist[nb] = dist[k] + 1; queue[tail++] = nb }
        }
        var seed = (g / 2) * g + g / 2
        if (!seedAll) for (k in 0 until g * g) if (dist[k] > dist[seed]) seed = k
        // Breadth-first phase unwrapping over the mask from the seed.
        val uu = DoubleArray(g * g) { Double.NaN }
        val vv = DoubleArray(g * g) { Double.NaN }
        uu[seed] = pu[seed]
        vv[seed] = pv[seed]
        head = 0
        tail = 0
        queue[tail++] = seed
        while (head < tail) {
            val k = queue[head++]
            for (nb in neighbours(k)) {
                if (nb < 0 || !mask[nb] || !uu[nb].isNaN()) continue
                uu[nb] = uu[k] + wrap(pu[nb] - pu[k] + 0.5, 1.0) - 0.5
                vv[nb] = vv[k] + wrap(pv[nb] - pv[k] + 1.0, 2.0) - 1.0
                queue[tail++] = nb
            }
        }
        var m = 0
        for (k in 0 until g * g) {
            if (uu[k].isNaN()) continue
            uv[2 * m] = uu[k]; uv[2 * m + 1] = vv[k]
            xy[2 * m] = (k % g) * STRIDE + STRIDE / 2 + 0.5
            xy[2 * m + 1] = (k / g) * STRIDE + STRIDE / 2 + 0.5
            m++
        }
        val r = HomographyFit.ransac(uv, xy, m, ransacPx(m), iterations = 500) ?: return null
        return if (r.inlierFraction >= MIN_INLIER_FRAC) r.h else null
    }

    private fun acquire(): DoubleArray? {
        if (n < ACQUIRE_MIN_AREA * g * g || n == g * g) return null   // a full crop has no edge to place it by
        val h0 = unwrappedFit() ?: return null
        val sc = offsetIous(h0)
        var best = 0
        for (k in sc.indices) if (sc[k] > sc[best]) best = k
        var second = -1.0
        for (k in sc.indices) if (k != best && sc[k] > second) second = sc[k]
        if (sc[best] < ACQUIRE_MIN_IOU || sc[best] - second < ACQUIRE_MARGIN) return null
        return Mat3.mul(h0, Mat3.translate(-offsetDu(best).toDouble(), -offsetDv(best).toDouble()))
    }

    private val nb4 = IntArray(4)
    private fun neighbours(k: Int): IntArray {
        val x = k % g
        val y = k / g
        nb4[0] = if (x > 0) k - 1 else -1
        nb4[1] = if (x < g - 1) k + 1 else -1
        nb4[2] = if (y > 0) k - g else -1
        nb4[3] = if (y < g - 1) k + g else -1
        return nb4
    }

    companion object {
        const val STRIDE = 4
        const val MIN_FIT_POINTS = 150
        const val MIN_INLIER_FRAC = 0.5
        const val RANSAC_PX = 4.0          // inlier threshold floor (far away)...
        const val RANSAC_CELLS = 0.06      // ...but at least this fraction of a cell: up close a cell is hundreds of px
        const val LOST_AFTER = 8
        const val LOST_AFTER_ON_PANEL = 60   // panel fills most of the view: still on it, coast on the gyro longer
        const val ON_PANEL_FRACTION = 0.6
        const val AGREE_CELLS = 0.15
        const val SHIFT_MARGIN = 0.04
        const val ACQUIRE_MIN_AREA = 0.05
        const val ACQUIRE_MIN_IOU = 0.6
        const val ACQUIRE_MARGIN = 0.08

        fun wrap(a: Double, period: Double): Double {
            val r = a - floor(a / period) * period
            return if (r >= period) r - period else r
        }
    }
}
