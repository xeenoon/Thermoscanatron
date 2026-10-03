package com.euhack.solar

import com.euhack.hello.CameraIntrinsics
import com.euhack.hello.ThermalPose
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

class PanelTrackerTest {
    private val spec = PanelSpec()
    private val size = 384

    /** What a perfect PanelNet would output for panel homography [h] (panel -> crop pixels), at [n] x [n]. */
    private fun dense(h: DoubleArray, n: Int = size): FloatArray {
        val size = n
        val plane = size * size
        val out = FloatArray(6 * plane)
        val inv = Mat3.inv(h)
        val uv = DoubleArray(2)
        for (y in 0 until size) for (x in 0 until size) {
            val w = Mat3.apply(inv, x + 0.5, y + 0.5, uv)
            val i = y * size + x
            if (w <= 0 || uv[0] < 0 || uv[0] > spec.cols || uv[1] < 0 || uv[1] > spec.rows) continue
            out[i] = 1f
            out[2 * plane + i] = sin(2 * PI * uv[0]).toFloat()
            out[3 * plane + i] = cos(2 * PI * uv[0]).toFloat()
            out[4 * plane + i] = sin(PI * uv[1]).toFloat()
            out[5 * plane + i] = cos(PI * uv[1]).toFloat()
        }
        return out
    }

    /** Panel with cells [cell] px wide, whose point (u0, v0) is at the crop centre. */
    private fun view(cell: Double, u0: Double, v0: Double) = doubleArrayOf(
        cell, 0.0, size / 2 - u0 * cell, 0.0, cell * 0.69, size / 2 - v0 * cell * 0.69, 0.0, 0.0, 1.0)

    @Test
    fun acquiresThenKeepsCellWhileZoomingIn() {
        val t = PanelTracker(spec, size)
        val id = listOf(Mat3.identity())
        // Far: whole panel in view.
        t.step(dense(view(30.0, 2.5, 4.5)), 1f, id)
        assertEquals(PanelTracker.State.ACQUIRED, t.state)
        assertArrayEquals(intArrayOf(4, 2), t.cellAt(size / 2.0, size / 2.0))
        // Zoom in on cell (6, 1) in steps; at the end one cell fills the crop and no edge is visible.
        var cell = 30.0
        var u = 2.5; var v = 4.5
        repeat(25) {
            cell *= 1.12
            u += (1.5 - u) * 0.2
            v += (6.5 - v) * 0.2
            val prevH = t.h!!
            val next = view(cell, u, v)
            // The true inter-frame motion is not given: identity prior only (worst case, no gyro).
            t.step(dense(next), 1f, id)
            assertNotNull("lost at zoom $cell", t.h)
            assertTrue(prevH !== t.h)
        }
        assertEquals(PanelTracker.State.TRACKING, t.state)
        assertArrayEquals(intArrayOf(6, 1), t.cellAt(size / 2.0, size / 2.0))
    }

    @Test
    fun relocksAfterAFastZoomIn() {
        // Far, then in one step 6x closer on the same spot (too far for the prediction): the centre cell holds.
        val t = PanelTracker(spec, size)
        val id = listOf(Mat3.identity())
        t.step(dense(view(30.0, 2.5, 4.5)), 1f, id)
        t.step(dense(view(180.0, 2.6, 4.4)), 1f, id)
        assertEquals(PanelTracker.State.RELOCKED, t.state)
        assertArrayEquals(intArrayOf(4, 2), t.cellAt(size / 2.0, size / 2.0))
    }

    @Test
    fun smallModelOutputAtHalfResolution() {
        // The fast path's model outputs 192 x 192 for the same 384-unit view: H scaled by 1/2 for rendering.
        val t = PanelTracker(spec, size)
        val h = view(30.0, 2.5, 4.5)
        val half = Mat3.mul(doubleArrayOf(0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0), h)
        t.step(dense(half, 192), 1f, listOf(Mat3.identity()), denseSize = 192)
        assertEquals(PanelTracker.State.ACQUIRED, t.state)
        assertArrayEquals(intArrayOf(4, 2), t.cellAt(size / 2.0, size / 2.0))
    }

    @Test
    fun hotspotFlagged() {
        // Thermal camera at the phone lens, looking straight ahead; panel 100 cm away, centred.
        val pose = ThermalPose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, false)
        val th = PanelThermal(spec, pose)
        val cam = CameraIntrinsics(460.0, 240.0, 320.0)
        val z = 100.0
        // Panel -> frame pixels: K [r1 r2 t] diag(cell w, cell h), t = top-left corner.
        val tx = -2 * spec.cellWcm; val ty = -4.5 * spec.cellHcm
        val h = doubleArrayOf(
            cam.f * spec.cellWcm, 0.0, cam.f * tx + cam.cx * z,
            0.0, cam.f * spec.cellHcm, cam.f * ty + cam.cy * z,
            0.0, 0.0, z)
        val plane = th.plane(h, cam)!!
        val ray = DoubleArray(3)
        val temps = FloatArray(768) { 15f }
        for (p in 0 until 768) {
            pose.ray((p % 32).toDouble(), (p / 32).toDouble(), ray)
            val s = z / ray[2]
            val u = (s * ray[0] - tx) / spec.cellWcm
            val v = (s * ray[1] - ty) / spec.cellHcm
            if (u.toInt() == 2 && v.toInt() == 4 && u > 0 && v > 0) temps[p] = 25f
        }
        th.update(temps, plane)
        assertTrue(th.cellFrames[4 * 4 + 2] > 0)
        assertEquals(25.0, th.cellC[4 * 4 + 2], 1e-6)
        assertTrue(th.isHotspot(4 * 4 + 2))
        assertEquals(1, (0 until 36).count { th.isHotspot(it) })
    }
}

