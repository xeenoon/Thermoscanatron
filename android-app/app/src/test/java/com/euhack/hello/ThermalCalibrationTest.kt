package com.euhack.hello

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Random
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt

/** The phone calibration must recover arbitrary mountings, like ML/tests/test_thermal_calib.py. */
class ThermalCalibrationTest {
    private val cam = CameraIntrinsics(460.0, 240.0, 320.0)
    private val box = intArrayOf(24, 104, 432)

    /** Ellipse "hands" at random places/depths, seen by a thermal camera at [truth], 8 Hz, 250 ms latency. */
    private fun synthetic(truth: DoubleArray, mirror: Boolean, seconds: Double, seed: Long,
                          zMin: Double = 18.0, zMax: Double = 45.0):
        kotlin.Pair<List<ThermalCalibration.CameraSample>, List<ThermalCalibration.ThermalSample>> {
        val rnd = Random(seed)
        val g = ThermalCalibration.MASK_GRID
        val rays = ThermalCalibration.rays(ThermalPose.of(truth, mirror))
        val cams = ArrayList<ThermalCalibration.CameraSample>()
        val thermals = ArrayList<ThermalCalibration.ThermalSample>()
        val frac = FloatArray(768)
        val valid = BooleanArray(768)
        val n = (seconds * 8).toInt()
        for (i in 0 until n) {
            val z = zMin + rnd.nextDouble() * (zMax - zMin)
            val r = cam.f * 5.5 / z / box[2] * g
            val gx = r + rnd.nextDouble() * (g - 2 * r)
            val gy = r + rnd.nextDouble() * (g - 2 * r)
            val mask = FloatArray(g * g) { k ->
                val dx = (k % g - gx) / r
                val dy = (k / g - gy) / (1.6 * r)
                if (dx * dx + dy * dy <= 1) 1f else 0f
            }
            val px = box[0] + (gx + 0.5) / g * box[2]
            val py = box[1] + (gy + 0.5) / g * box[2]
            val ts = i * 125_000_000L
            val area = mask.count { it > 0.5f }.toDouble() / mask.size * box[2] * box[2]
            val sample = ThermalCalibration.CameraSample(ts, mask, box[0], box[1], box[2], z,
                doubleArrayOf((px - cam.cx) / cam.f * z, (py - cam.cy) / cam.f * z, z), area)
            cams.add(sample)
            ThermalCalibration.predict(rays, truth, sample, cam, frac, valid)
            val celsius = FloatArray(768) { k ->
                (21 + 12 * (if (valid[k]) frac[k] else 0f) + rnd.nextGaussian() * 0.3).toFloat()
            }
            thermals.add(ThermalCalibration.ThermalSample(ts + 250_000_000L, celsius))
        }
        return kotlin.Pair(cams, thermals)
    }

    private fun check(deg: DoubleArray, cm: DoubleArray, mirror: Boolean, seconds: Double,
                      zMin: Double = 18.0, zMax: Double = 45.0, cmTol: Double = 1.5, degTol: Double = 1.5) {
        val truth = doubleArrayOf(Math.toRadians(deg[0]), Math.toRadians(deg[1]), Math.toRadians(deg[2]),
            cm[0], cm[1], cm[2], ln(1.05))
        val (cams, thermals) = synthetic(truth, mirror, seconds, 1, zMin, zMax)
        val r = ThermalCalibration.solve(cams, thermals, cam)
        assertNotNull("too few pairs", r)
        r!!
        val p = r.pose.toArray()
        assertEquals(mirror, r.pose.mirror)
        for (i in 0..2) {
            var d = Math.toDegrees(p[i] - truth[i]) % 360
            if (d > 180) d -= 360
            if (d < -180) d += 360
            assertTrue("angle $i off by $d deg", abs(d) < degTol)
        }
        for (i in 3..5) assertTrue("offset ${i - 3} ${p[i]} vs ${truth[i]}", abs(p[i] - truth[i]) < cmTol)
        // The lens scale is weakly observable and carries a datasheet prior: only check it stays sane.
        assertTrue("focal scale ${exp(p[6])}", abs(exp(p[6]) - 1.05) < 0.06)
        // Measured from hand-size changes, or the default when there are too few (it pairs the same frames).
        assertTrue("latency ${r.latencyMs}", r.latencyMs == 250 || !r.latencyMeasured)
    }

    @Test fun recoversYaw30Roll45Offset20cm() = check(doubleArrayOf(-30.0, 5.0, 45.0), doubleArrayOf(20.0, -3.0, 1.0), true, 11.0)

    @Test fun recoversNearPhoneMounting() = check(doubleArrayOf(2.0, -3.0, 92.0), doubleArrayOf(0.0, 3.5, -1.0), true, 11.0)

    @Test fun recoversUnmirroredSteepMounting() = check(doubleArrayOf(10.0, 20.0, -60.0), doubleArrayOf(-8.0, 10.0, 0.0), false, 11.0)

    /**
     * The in-app case that produced a facing-backwards "twin": 5 s with the hand at nearly one depth. The
     * points are near-coplanar, so only the physical constraints pick the real rig; depth is barely
     * observable there, so offsets get a looser tolerance.
     */
    @Test fun fiveSecondsAtOneDepthIsNotTheTwin() =
        check(doubleArrayOf(2.0, -3.0, 95.0), doubleArrayOf(0.0, 3.5, -1.0), true, 5.0, 23.0, 27.0, cmTol = 4.0, degTol = 4.0)

    /**
     * 5-second slices of the recorded session must find the mounting (mirror, ~95 deg roll); those with the
     * near/far depth spread the app requires must also map the hand within 2.5 thermal px (~4 deg) of the
     * full fit: 5 s of hand only partly separates tilt from offset, the roll and mirroring are robust. Tilt vs offset is not
     * separable from such a slice, which is why the app asks for the hand near and far.
     */
    @Test fun fiveSecondSlicesOfRecordedSession() {
        val dir = File("../../ML/data/thermal_sessions/20261002_145939")
        assumeTrue("session not present", dir.isDirectory)
        val (cams, thermals, intr) = loadSession(dir)
        val full = ThermalCalibration.solve(cams.filter { usable(it.tsNs - cams.minOf { c -> c.tsNs }) }, thermals, intr)!!.pose
        val t0 = cams.minOf { it.tsNs }
        val a = DoubleArray(2)
        val b = DoubleArray(2)
        for (start in listOf(5.0, 20.0, 35.0, 48.0)) {
            val sel = cams.filter { ((it.tsNs - t0) / 1e9) in start..(start + 5) }
            val r = ThermalCalibration.solve(sel, thermals, intr)!!
            val p = r.pose.toArray()
            var worst = 0.0
            for (x in 100..380 step 70) for (y in 140..500 step 90) {
                val z = 25.0
                val px = (x - intr.cx) / intr.f * z
                val py = (y - intr.cy) / intr.f * z
                full.project(px, py, z, a)
                r.pose.project(px, py, z, b)
                if (a[0] in 0.0..31.0 && a[1] in 0.0..23.0) worst = maxOf(worst, kotlin.math.hypot(a[0] - b[0], a[1] - b[1]))
            }
            println("slice ${start}s: mirror=${r.pose.mirror} ypr ${(0..2).map { "%.1f".format(Math.toDegrees(p[it])) }} " +
                "xyz ${p.slice(3..5).map { "%.1f".format(it) }} k ${"%.2f".format(exp(p[6]))} corr ${"%.2f".format(r.correlation)} " +
                "n ${r.pairs} worst at 25 cm ${"%.2f".format(worst)} px")
            val spread = ThermalCalibration.depthSpread(sel)
            println("    depth spread ${"%.2f".format(spread)}")
            assertTrue(r.pose.mirror)
            assertEquals(95.0, Math.toDegrees(p[2]), 6.0)
            // Only captures the app would accept (hand near and far) must agree closely; single-depth slices
            // cannot tell a tilt from an offset (printed above: up to ~3 px at hand depth).
            if (spread >= ThermalCalibration.MIN_DEPTH_SPREAD) assertTrue("slice $start: $worst px", worst < 2.5)
        }
    }

    /** In-app captures copied to ML/data/calibrations/ (adb pull .../files/calibrations): replay and print. */
    @Test fun replayInAppCaptures() {
        val dir = File("../../ML/data/calibrations")
        val files = dir.listFiles { f -> f.name.endsWith(".bin") }?.sorted() ?: emptyList()
        assumeTrue("no captures", files.isNotEmpty())
        for (f in files) {
            val (cams, thermals, intr) = ThermalCalibration.readCapture(f)
            val lag = ThermalCalibration.estimateLatency(cams, thermals)
            val r = ThermalCalibration.solve(cams, thermals, intr)
            println("${f.name}: ${cams.size} cams, ${thermals.size} thermal, spread ${"%.2f".format(ThermalCalibration.depthSpread(cams))}, lag $lag")
            if (r != null) println(CalibrationStore.describe(r))
        }
    }

    /**
     * Capture 20261003_160305 (hand at 23-29 cm only) once fitted 6.5 cm right, 8.5 cm down, 4.7 cm behind the
     * lens with a compensating 27 deg tilt: right at 25 cm, ~25 deg up-left at 1.5 m. With the taped-rig limits
     * and the zero-offset start it must stay near the phone axis, and fit at least as well.
     */
    @Test fun singleDepthCaptureDoesNotTradeTiltForOffset() {
        val f = File("../../ML/data/calibrations/capture_20261003_160305.bin")
        assumeTrue("capture not present", f.isFile)
        val (cams, thermals, intr) = ThermalCalibration.readCapture(f)
        val r = ThermalCalibration.solve(cams, thermals, intr, rig = ThermalCalibration.Rig.TAPED)!!
        val p = r.pose.toArray()
        println(CalibrationStore.describe(r))
        assertTrue(r.pose.mirror)
        assertTrue("yaw ${Math.toDegrees(p[0])}", abs(Math.toDegrees(p[0])) < 10)
        assertTrue("pitch ${Math.toDegrees(p[1])}", abs(Math.toDegrees(p[1])) < 10)
        assertTrue("offset ${p.slice(3..5)}", sqrt(p[3] * p[3] + p[4] * p[4] + p[5] * p[5]) < 6)
        assertTrue("correlation ${r.correlation}", r.correlation > 0.61)
    }

    private fun usable(dtNs: Long) = (dtNs / 1e9).let { it in 0.0..55.0 || it in 60.0..70.0 }

    private fun loadSession(dir: File): Triple<List<ThermalCalibration.CameraSample>, List<ThermalCalibration.ThermalSample>, CameraIntrinsics> {
        val intr = CameraIntrinsics(4.69 / 6.528 * 4080 * 0.1568627506494522, 240.0, 320.0)
        val rows = File(dir, "frames.csv").readLines().drop(1).map { it.split(",") }
        val cams = rows.filter { it[10] == "1" }.mapNotNull { r ->
            val mask = readGreyPng(File(dir, "masks/%06d.png".format(r[0].toInt())))
            ThermalCalibration.cameraSample(r[1].toLong(), mask, 384, r[6].toInt(), r[7].toInt(), r[8].toInt(), intr)
        }
        val raw = File(dir, "thermal.bin").readBytes()
        val bb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN)
        val thermals = (0 until raw.size / 1574).map { i ->
            val o = i * 1574
            ThermalCalibration.ThermalSample(bb.getLong(o), FloatArray(768) { bb.getShort(o + 36 + 2 * it) / 100f })
        }
        return Triple(cams, thermals, intr)
    }

    /** javax.imageio is on the test JVM but not in android.jar, so it is reached by reflection. */
    private fun readGreyPng(file: File): FloatArray {
        val img = Class.forName("javax.imageio.ImageIO").getMethod("read", File::class.java).invoke(null, file)
        val w = img.javaClass.getMethod("getWidth").invoke(img) as Int
        val h = img.javaClass.getMethod("getHeight").invoke(img) as Int
        val rgb = IntArray(w * h)
        img.javaClass.getMethod("getRGB", Int::class.java, Int::class.java, Int::class.java, Int::class.java,
            IntArray::class.java, Int::class.java, Int::class.java).invoke(img, 0, 0, w, h, rgb, 0, w)
        return FloatArray(w * h) { (rgb[it] and 0xFF) / 255f }
    }

    /** The recorded session (ML/data, not in git): the phone solver should agree with the Python fit. */
    @Test fun matchesPythonOnRecordedSession() {
        val dir = File("../../ML/data/thermal_sessions/20261002_145939")
        assumeTrue("session not present", dir.isDirectory)
        val f = 4.69 / 6.528 * 4080 * 0.1568627506494522
        val intr = CameraIntrinsics(f, 240.0, 320.0)
        val rows = File(dir, "frames.csv").readLines().drop(1).map { it.split(",") }
        val t0 = rows[0][1].toLong()
        fun usable(ts: Long) = ((ts - t0) / 1e9).let { it in 0.0..55.0 || it in 60.0..70.0 }
        val cams = rows.filter { it[10] == "1" && usable(it[1].toLong()) }.mapNotNull { r ->
            val mask = readGreyPng(File(dir, "masks/%06d.png".format(r[0].toInt())))
            ThermalCalibration.cameraSample(r[1].toLong(), mask, 384, r[6].toInt(), r[7].toInt(), r[8].toInt(), intr)
        }
        val raw = File(dir, "thermal.bin").readBytes()
        val bb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN)
        val thermals = (0 until raw.size / 1574).map { i ->
            val o = i * 1574
            ThermalCalibration.ThermalSample(bb.getLong(o), FloatArray(768) { bb.getShort(o + 36 + 2 * it) / 100f })
        }
        val r = ThermalCalibration.solve(cams, thermals, intr)!!
        val p = r.pose.toArray()
        println("phone solver: mirror=${r.pose.mirror} yaw/pitch/roll ${(0..2).map { Math.toDegrees(p[it]) }} " +
            "xyz ${p.slice(3..5)} k ${exp(p[6])} latency ${r.latencyMs} corr ${r.correlation} pairs ${r.pairs}")
        // Python (runs/thermal_calib, measured 49.7 x 38.3 deg field of view): yaw -4.3, pitch 6.6, roll 97.9 deg,
        // x -0.2, y 5.2, z -0.3 cm, corr 0.830. The correlation peak is flat to ~2-3 deg / ~1.5 cm with hands at
        // 20-30 cm, so agree to that.
        assertTrue(r.pose.mirror)
        assertEquals(-4.3, Math.toDegrees(p[0]), 3.0)
        assertEquals(6.6, Math.toDegrees(p[1]), 3.0)
        assertEquals(97.9, Math.toDegrees(p[2]), 2.0)
        assertEquals(5.2, p[4], 2.0)
        assertTrue(r.correlation > 0.80)
    }
}
