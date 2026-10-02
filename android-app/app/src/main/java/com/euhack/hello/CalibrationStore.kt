package com.euhack.hello

import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.Locale

/** The last accepted camera <-> thermal calibration, persisted as <files>/thermal_calib.json. */
class CalibrationStore(dir: File) {
    private val file = File(dir, "thermal_calib.json")

    fun load(): ThermalCalibration.Result? = try {
        if (!file.isFile) null else {
            val o = JSONObject(file.readText())
            val p = o.getJSONArray("pose")
            val s = o.getJSONArray("sigma")
            ThermalCalibration.Result(
                ThermalPose.of(DoubleArray(7) { p.getDouble(it) }, o.getBoolean("mirror")),
                o.getInt("latency_ms"), o.getDouble("correlation"), o.getDouble("stage1_px"),
                DoubleArray(7) { s.optDouble(it, Double.NaN) }, o.getInt("pairs"),
            )
        }
    } catch (e: Exception) {
        null
    }

    fun save(r: ThermalCalibration.Result) {
        val p = r.pose.toArray()
        file.writeText(JSONObject()
            .put("pose", JSONArray(p.toList()))
            .put("mirror", r.pose.mirror)
            .put("latency_ms", r.latencyMs)
            .put("correlation", r.correlation)
            .put("stage1_px", r.stage1Px)
            .put("sigma", JSONArray(r.sigma.map { if (it.isFinite()) it else JSONObject.NULL }))
            .put("pairs", r.pairs)
            .put("readable", describe(r))
            .toString(2))
    }

    companion object {
        /** The popup text: offsets in degrees and centimetres with rough 1-sigma. */
        fun describe(r: ThermalCalibration.Result): String {
            val p = r.pose.toArray()
            val s = r.sigma
            fun deg(i: Int) = String.format(Locale.US, "%+.1f°  ± %.1f", Math.toDegrees(p[i]), Math.toDegrees(s[i]))
            fun cm(i: Int) = String.format(Locale.US, "%+.1f cm  ± %.1f", p[i], s[i])
            return """
                |Thermal camera relative to the phone camera:
                |
                |Yaw (turned right)   ${deg(0)}
                |Pitch (tilted up)    ${deg(1)}
                |Roll (clockwise)     ${deg(2)}
                |
                |X (right)            ${cm(3)}
                |Y (down)             ${cm(4)}
                |Z (forward)          ${cm(5)}
                |
                |Mirrored sensor: ${if (r.pose.mirror) "yes" else "no"}   lens scale ${String.format(Locale.US, "%.2f", Math.exp(p[6]))}
                |Fit: correlation ${String.format(Locale.US, "%.2f", r.correlation)} over ${r.pairs} frames, thermal lag ${r.latencyMs} ms
            """.trimMargin()
        }
    }
}
