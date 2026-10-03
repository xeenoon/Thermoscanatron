package com.euhack.hello

import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File

/** Where does calibration time go? Replays an in-app capture stage by stage (desktop JVM, single thread). */
class SolverProfileTest {
    @Test fun profile() {
        val f = File("../../ML/data/calibrations/capture_20261002_162751.bin")
        assumeTrue(f.isFile)
        val (cams, thermals, cam) = ThermalCalibration.readCapture(f)
        fun <T> timed(name: String, block: () -> T): T {
            val t = System.nanoTime(); val r = block()
            println("PROFILE %-28s %7.0f ms".format(name, (System.nanoTime() - t) / 1e6)); return r
        }
        timed("warm-up solve") { ThermalCalibration.solve(cams, thermals, cam) }
        val lat = timed("stage 0 latency") { ThermalCalibration.estimateLatency(cams, thermals) }
        val pairs = timed("pairing") { ThermalCalibration.makePairs(cams, thermals, lat?.first ?: 225) }
        val (x1, mirror, _) = timed("stage 1 multi-start (600 LM)") { ThermalCalibration.poseFromPoints(pairs) }
        val p1 = doubleArrayOf(x1[0], x1[1], x1[2], x1[3], x1[4], x1[5], 0.0)
        var evals = 0
        timed("one correlation eval") { ThermalCalibration.correlation(p1, mirror, pairs, cam) }
        val step = doubleArrayOf(0.05, 0.05, 0.05, 2.0, 2.0, 2.0, 0.1)
        timed("stage 2 Nelder-Mead") {
            ThermalCalibration.nelderMead({ q -> evals++; -ThermalCalibration.correlation(q, mirror, pairs, cam) }, p1, step, 800)
        }
        println("PROFILE stage 2 evaluations: $evals, pairs ${pairs.size}")
        val r = timed("full solve()") { ThermalCalibration.solve(cams, thermals, cam) }!!
        val q = r.pose.toArray()
        println("PROFILE result ypr ${(0..2).map { "%.2f".format(Math.toDegrees(q[it])) }} xyz ${q.slice(3..5).map { "%.2f".format(it) }} " +
            "k ${"%.3f".format(Math.exp(q[6]))} corr ${"%.4f".format(r.correlation)}")
    }
}
