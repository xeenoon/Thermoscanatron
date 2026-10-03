package com.euhack.solar

import java.util.Random
import kotlin.math.abs
import kotlin.math.sqrt

/** 3x3 matrices as row-major DoubleArray(9), and homography fitting (DLT + RANSAC). */
object Mat3 {
    fun identity() = doubleArrayOf(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    fun mul(a: DoubleArray, b: DoubleArray): DoubleArray {
        val o = DoubleArray(9)
        for (r in 0 until 3) for (c in 0 until 3) {
            o[r * 3 + c] = a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c]
        }
        return o
    }

    fun inv(m: DoubleArray): DoubleArray {
        val a = m[0]; val b = m[1]; val c = m[2]
        val d = m[3]; val e = m[4]; val f = m[5]
        val g = m[6]; val h = m[7]; val i = m[8]
        val A = e * i - f * h; val B = -(d * i - f * g); val C = d * h - e * g
        val det = a * A + b * B + c * C
        val k = 1.0 / det
        return doubleArrayOf(
            A * k, -(b * i - c * h) * k, (b * f - c * e) * k,
            B * k, (a * i - c * g) * k, -(a * f - c * d) * k,
            C * k, -(a * h - b * g) * k, (a * e - b * d) * k,
        )
    }

    fun translate(dx: Double, dy: Double) = doubleArrayOf(1.0, 0.0, dx, 0.0, 1.0, dy, 0.0, 0.0, 1.0)

    /** (x, y) -> H (x, y, 1), dehomogenised into out; returns the w before division. */
    fun apply(h: DoubleArray, x: Double, y: Double, out: DoubleArray): Double {
        val w = h[6] * x + h[7] * y + h[8]
        out[0] = (h[0] * x + h[1] * y + h[2]) / w
        out[1] = (h[3] * x + h[4] * y + h[5]) / w
        return w
    }

    /** Rotation matrix from a rotation vector (Rodrigues). */
    fun rotation(wx: Double, wy: Double, wz: Double): DoubleArray {
        val th = sqrt(wx * wx + wy * wy + wz * wz)
        if (th < 1e-12) return identity()
        val x = wx / th; val y = wy / th; val z = wz / th
        val c = Math.cos(th); val s = Math.sin(th); val t = 1 - c
        return doubleArrayOf(
            t * x * x + c, t * x * y - s * z, t * x * z + s * y,
            t * x * y + s * z, t * y * y + c, t * y * z - s * x,
            t * x * z - s * y, t * y * z + s * x, t * z * z + c,
        )
    }

    fun transpose(m: DoubleArray) = doubleArrayOf(m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8])
}

object HomographyFit {
    private const val SCORE_POINTS = 400
    private const val EARLY_EXIT = 0.85   // a hypothesis this good ends the search

    class Result(val h: DoubleArray, val inlierFraction: Double)

    /**
     * RANSAC homography mapping src (x, y pairs) to dst, inlier threshold [thresholdPx] in dst units, refit on all
     * inliers. Null if no non-degenerate model was found.
     */
    fun ransac(src: DoubleArray, dst: DoubleArray, n: Int, thresholdPx: Double, iterations: Int = 200,
               rng: Random = Random(1)): Result? {
        if (n < 8) return null
        val t2 = thresholdPx * thresholdPx
        var bestCount = 0
        var best: DoubleArray? = null
        val idx = IntArray(4)
        val out = DoubleArray(2)
        // Hypotheses are scored on a random subset (cheap on the phone); the winner is refit on all points.
        val score = IntArray(minOf(n, SCORE_POINTS)) { if (n <= SCORE_POINTS) it else rng.nextInt(n) }
        for (iter in 0 until iterations) {
            for (k in 0 until 4) {
                var j: Int
                do { j = rng.nextInt(n) } while ((0 until k).any { idx[it] == j })
                idx[k] = j
            }
            val h = fit(src, dst, idx, 4) ?: continue
            var count = 0
            for (i in score) {
                val w = Mat3.apply(h, src[2 * i], src[2 * i + 1], out)
                if (w <= 0) continue
                val dx = out[0] - dst[2 * i]
                val dy = out[1] - dst[2 * i + 1]
                if (dx * dx + dy * dy < t2) count++
            }
            if (count > bestCount) {
                bestCount = count
                best = h
            }
            if (bestCount > EARLY_EXIT * score.size) break
        }
        var h = best ?: return null
        // Refit on the inliers, then count again.
        repeat(2) {
            val inl = ArrayList<Int>()
            for (i in 0 until n) {
                val w = Mat3.apply(h, src[2 * i], src[2 * i + 1], out)
                val dx = out[0] - dst[2 * i]
                val dy = out[1] - dst[2 * i + 1]
                if (w > 0 && dx * dx + dy * dy < t2) inl.add(i)
            }
            if (inl.size >= 4) fit(src, dst, inl.toIntArray(), inl.size)?.let { h = it }
        }
        var count = 0
        for (i in 0 until n) {
            val w = Mat3.apply(h, src[2 * i], src[2 * i + 1], out)
            val dx = out[0] - dst[2 * i]
            val dy = out[1] - dst[2 * i + 1]
            if (w > 0 && dx * dx + dy * dy < t2) count++
        }
        return Result(h, count.toDouble() / n)
    }

    /** Normalised DLT with h33 = 1, least squares over the chosen points. */
    fun fit(src: DoubleArray, dst: DoubleArray, idx: IntArray, m: Int): DoubleArray? {
        val ns = normaliser(src, idx, m)
        val nd = normaliser(dst, idx, m)
        val ata = DoubleArray(64)
        val atb = DoubleArray(8)
        val row = DoubleArray(8)
        for (k in 0 until m) {
            val i = idx[k]
            val x = (src[2 * i] - ns[0]) * ns[2]
            val y = (src[2 * i + 1] - ns[1]) * ns[2]
            val u = (dst[2 * i] - nd[0]) * nd[2]
            val v = (dst[2 * i + 1] - nd[1]) * nd[2]
            // u = (h0 x + h1 y + h2) / (h6 x + h7 y + 1)
            row[0] = x; row[1] = y; row[2] = 1.0; row[3] = 0.0; row[4] = 0.0; row[5] = 0.0
            row[6] = -u * x; row[7] = -u * y
            accumulate(ata, atb, row, u)
            row[0] = 0.0; row[1] = 0.0; row[2] = 0.0; row[3] = x; row[4] = y; row[5] = 1.0
            row[6] = -v * x; row[7] = -v * y
            accumulate(ata, atb, row, v)
        }
        val p = solve(ata, atb, 8) ?: return null
        val hn = doubleArrayOf(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 1.0)
        // Undo the normalisation: H = Nd^-1 Hn Ns.
        val nsM = doubleArrayOf(ns[2], 0.0, -ns[0] * ns[2], 0.0, ns[2], -ns[1] * ns[2], 0.0, 0.0, 1.0)
        val ndInv = doubleArrayOf(1 / nd[2], 0.0, nd[0], 0.0, 1 / nd[2], nd[1], 0.0, 0.0, 1.0)
        val h = Mat3.mul(ndInv, Mat3.mul(hn, nsM))
        return if (h.all { it.isFinite() }) h else null
    }

    private fun accumulate(ata: DoubleArray, atb: DoubleArray, row: DoubleArray, b: Double) {
        for (r in 0 until 8) {
            if (row[r] == 0.0) continue
            for (c in 0 until 8) ata[r * 8 + c] += row[r] * row[c]
            atb[r] += row[r] * b
        }
    }

    /** (centroid x, centroid y, scale) so the points have mean distance sqrt(2) from the origin. */
    private fun normaliser(p: DoubleArray, idx: IntArray, m: Int): DoubleArray {
        var cx = 0.0; var cy = 0.0
        for (k in 0 until m) { cx += p[2 * idx[k]]; cy += p[2 * idx[k] + 1] }
        cx /= m; cy /= m
        var d = 0.0
        for (k in 0 until m) d += Math.hypot(p[2 * idx[k]] - cx, p[2 * idx[k] + 1] - cy)
        d /= m
        return doubleArrayOf(cx, cy, if (d > 1e-12) Math.sqrt(2.0) / d else 1.0)
    }

    /** Gaussian elimination with partial pivoting; null if singular. */
    private fun solve(a: DoubleArray, b: DoubleArray, n: Int): DoubleArray? {
        val m = a.copyOf()
        val x = b.copyOf()
        for (col in 0 until n) {
            var piv = col
            for (r in col + 1 until n) if (abs(m[r * n + col]) > abs(m[piv * n + col])) piv = r
            if (abs(m[piv * n + col]) < 1e-12) return null
            if (piv != col) {
                for (c in 0 until n) { val t = m[col * n + c]; m[col * n + c] = m[piv * n + c]; m[piv * n + c] = t }
                val t = x[col]; x[col] = x[piv]; x[piv] = t
            }
            for (r in col + 1 until n) {
                val f = m[r * n + col] / m[col * n + col]
                if (f == 0.0) continue
                for (c in col until n) m[r * n + c] -= f * m[col * n + c]
                x[r] -= f * x[col]
            }
        }
        for (r in n - 1 downTo 0) {
            var s = x[r]
            for (c in r + 1 until n) s -= m[r * n + c] * x[c]
            x[r] = s / m[r * n + r]
        }
        return x
    }
}
