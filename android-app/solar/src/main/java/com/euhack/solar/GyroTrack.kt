package com.euhack.solar

import android.hardware.SensorEvent

/**
 * Integrated gyroscope rotation with timestamps (SensorEvent.timestamp: elapsedRealtimeNanos, the camera's
 * clock), so the rotation between any two moments, e.g. two camera frames, or the last frame and now for the
 * 60 Hz overlay, can be read back. Small-angle sums: fine over the fraction of a second it is used for.
 * Axes are converted to the upright camera frame: x right, y down, z into the scene.
 */
class GyroTrack {
    private val lock = Any()
    private val t = LongArray(N)
    private val r = Array(N) { DoubleArray(3) }
    private var head = 0
    private var count = 0
    private val total = DoubleArray(3)
    private var lastNs = 0L
    @Volatile var rateRadS = 0.0
        private set

    fun onEvent(e: SensorEvent) = synchronized(lock) {
        if (lastNs != 0L) {
            val dt = (e.timestamp - lastNs) / 1e9
            if (dt in 0.0..0.1) {
                // Device axes (x right, y up, z out of the screen) -> camera axes (x right, y down, z forward).
                total[0] += e.values[0] * dt
                total[1] -= e.values[1] * dt
                total[2] -= e.values[2] * dt
                rateRadS = Math.sqrt((e.values[0] * e.values[0] + e.values[1] * e.values[1] +
                    e.values[2] * e.values[2]).toDouble())
            }
        }
        lastNs = e.timestamp
        t[head] = e.timestamp
        total.copyInto(r[head])
        head = (head + 1) % N
        if (count < N) count++
    }

    /** Integrated rotation vector at time [ns] (nearest sample at or before it). */
    fun at(ns: Long): DoubleArray = synchronized(lock) {
        var best = -1
        for (k in 0 until count) {
            val i = (head - 1 - k + N) % N
            if (t[i] <= ns) { best = i; break }
        }
        if (best < 0) total.copyOf() else r[best].copyOf()
    }

    fun now(): DoubleArray = synchronized(lock) { total.copyOf() }

    /** Rotation vector from [a] to [b] (camera axes, radians). */
    fun between(a: DoubleArray, b: DoubleArray) = doubleArrayOf(b[0] - a[0], b[1] - a[1], b[2] - a[2])

    /** Mean angular speed between two moments, rad/s. */
    fun speed(a: Long, b: Long): Double {
        if (b <= a) return 0.0
        val d = between(at(a), at(b))
        return Math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) / ((b - a) / 1e9)
    }

    companion object {
        private const val N = 2048   // ~4 s at 500 Hz
    }
}
