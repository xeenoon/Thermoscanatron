package com.euhack.hello

import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.hypot
import kotlin.math.sin

/** Phone camera as a pinhole, in pixels of the upright analysis frame (x right, y down). */
data class CameraIntrinsics(val f: Double, val cx: Double, val cy: Double)

/**
 * Where the thermal camera sits relative to the phone camera, plus its lens scale. Same model as
 * ML/src/segkit/thermal_calib.py:
 *  - camera coordinates: upright frame, x right, y down, z forward, centimetres;
 *  - thermal axes in camera coordinates R = Ry(yaw) Rx(pitch) Rz(roll): yaw + turns the thermal camera
 *    right, pitch + up, roll + clockwise as seen from behind the phone;
 *  - thermal centre at (x, y, z) cm;
 *  - thermal lens: MLX90640ESF-BAB, equidistant (angle from axis = radius / f), focal lengths from the
 *    measured field of view ([ThermalGeometry.FOV_X_DEG] x [ThermalGeometry.FOV_Y_DEG]) times exp(logK);
 *    [mirror] flips the sensor's x axis (datasheet Figure 3: column 1 is on the right, looking out).
 */
class ThermalPose(
    val yaw: Double, val pitch: Double, val roll: Double,
    val x: Double, val y: Double, val z: Double,
    val logK: Double,
    val mirror: Boolean,
) {
    private val r = ThermalGeometry.rotation(yaw, pitch, roll)
    private val fx = ThermalGeometry.FX * exp(logK)
    private val fy = ThermalGeometry.FY * exp(logK)
    private val sx = if (mirror) -1.0 else 1.0

    fun toArray() = doubleArrayOf(yaw, pitch, roll, x, y, z, logK)

    /** Camera-frame point (cm) -> thermal pixel (u = column, v = row) in [out]. */
    fun project(px: Double, py: Double, pz: Double, out: DoubleArray) {
        val dx = px - x
        val dy = py - y
        val dz = pz - z
        // q = R^T (P - c)
        val qx = r[0] * dx + r[3] * dy + r[6] * dz
        val qy = r[1] * dx + r[4] * dy + r[7] * dz
        val qz = r[2] * dx + r[5] * dy + r[8] * dz
        val rr = hypot(qx, qy)
        val s = if (rr > 1e-9) atan2(rr, qz) / rr else 0.0
        out[0] = ThermalGeometry.CX + sx * fx * s * qx
        out[1] = ThermalGeometry.CY + fy * s * qy
    }

    /** Unit ray through thermal pixel (u, v), in camera coordinates, into [out]. */
    fun ray(u: Double, v: Double, out: DoubleArray) {
        val mx = (u - ThermalGeometry.CX) / fx * sx
        val my = (v - ThermalGeometry.CY) / fy
        val th = hypot(mx, my)
        val s = if (th > 1e-9) sin(th) / th else 1.0
        val tx = mx * s
        val ty = my * s
        val tz = cos(th)
        out[0] = r[0] * tx + r[1] * ty + r[2] * tz
        out[1] = r[3] * tx + r[4] * ty + r[5] * tz
        out[2] = r[6] * tx + r[7] * ty + r[8] * tz
    }

    companion object {
        fun of(p: DoubleArray, mirror: Boolean) = ThermalPose(p[0], p[1], p[2], p[3], p[4], p[5], p[6], mirror)
    }
}

object ThermalGeometry {
    const val W = 32
    const val H = 24
    const val CX = (W - 1) / 2.0
    const val CY = (H - 1) / 2.0
    /**
     * Field of view MEASURED on this sensor with a soldering-iron point source (session 20261002_171833,
     * 126 pairs, bootstrap 95%): 49.7 (49.3-50.0) x 38.3 (37.9-39.0) deg. The datasheet's "typical" 55 x 35
     * (Table 15) is defined at the 50%-sensitivity edge of the whole array, not by pixel-centre spacing.
     */
    const val FOV_X_DEG = 49.7
    const val FOV_Y_DEG = 38.3
    val FX = (W / 2) / Math.toRadians(FOV_X_DEG / 2)   // px per radian, horizontal (32 px)
    val FY = (H / 2) / Math.toRadians(FOV_Y_DEG / 2)   // px per radian, vertical (24 px)

    /** Row-major 3x3 Ry(yaw) Rx(pitch) Rz(roll). */
    fun rotation(yaw: Double, pitch: Double, roll: Double): DoubleArray {
        val cy = cos(yaw); val sy = sin(yaw)
        val cp = cos(pitch); val sp = sin(pitch)
        val cr = cos(roll); val sr = sin(roll)
        // Ry * Rx
        val a = doubleArrayOf(cy, sy * sp, sy * cp, 0.0, cp, -sp, -sy, cy * sp, cy * cp)
        // (Ry Rx) * Rz
        return doubleArrayOf(
            a[0] * cr + a[1] * sr, -a[0] * sr + a[1] * cr, a[2],
            a[3] * cr + a[4] * sr, -a[3] * sr + a[4] * cr, a[5],
            a[6] * cr + a[7] * sr, -a[6] * sr + a[7] * cr, a[8],
        )
    }
}
