package com.euhack.hello

import kotlin.math.abs

/**
 * Image translation between consecutive model crops by exhaustive SAD block matching on a 1/4-scale grey copy
 * (port of block_shift in ML/src/segkit/panel/track.py). Shared by both apps: the panel tracker coasts on it up
 * close (inside one cell every spot looks alike), the hand app carries the big model's late masks forward with it.
 */
class BlockMotion(private val size: Int, private val scale: Int = SCALE) {
    private val n = size / scale
    private var prev: FloatArray? = null
    private val cur = FloatArray(n * n)

    /** Feed the crop's ARGB pixels (size x size); returns (dx, dy) in those pixels since the previous crop, or null
     *  at the start. */
    fun update(argb: IntArray): DoubleArray? {
        var mean = 0f
        for (y in 0 until n) for (x in 0 until n) {
            var s = 0f
            for (j in 0 until scale) for (i in 0 until scale) {
                val p = argb[(y * scale + j) * size + x * scale + i]
                s += ((p shr 16) and 0xFF) * 0.299f + ((p shr 8) and 0xFF) * 0.587f + (p and 0xFF) * 0.114f
            }
            cur[y * n + x] = s / (scale * scale)
            mean += cur[y * n + x]
        }
        mean /= n * n
        for (k in cur.indices) cur[k] -= mean
        val a = prev
        val out = if (a == null) null else {
            val m = RADIUS
            var best = Float.MAX_VALUE
            var bx = 0
            var by = 0
            for (dy in -m..m) for (dx in -m..m) {
                var c = 0f
                for (y in m until n - m) {
                    val ra = y * n
                    val rb = (y + dy) * n + dx
                    for (x in m until n - m) c += abs(cur[rb + x] - a[ra + x])
                    if (c >= best) break
                }
                if (c < best) { best = c; bx = dx; by = dy }
            }
            doubleArrayOf(bx * scale.toDouble(), by * scale.toDouble())
        }
        prev = (prev ?: FloatArray(n * n)).also { cur.copyInto(it) }
        return out
    }

    fun reset() { prev = null }

    companion object {
        const val SCALE = 4
        const val RADIUS = 8   // +-8 px at 1/4 scale = +-32 crop px per frame
    }
}
