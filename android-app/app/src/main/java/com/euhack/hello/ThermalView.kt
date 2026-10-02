package com.euhack.hello

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.view.View

/**
 * Full-screen thermal image: the 32x24 frame, colour-mapped over its 2nd-98th percentile range and
 * scaled up with bilinear filtering (FIT_CENTER). The hottest pixel is marked with a ring.
 */
class ThermalView(context: Context) : View(context) {
    private val bitmap = Bitmap.createBitmap(ThermalPacketParser.WIDTH, ThermalPacketParser.HEIGHT, Bitmap.Config.ARGB_8888)
    private val colors = IntArray(ThermalPacketParser.PIXELS)
    private val bitmapPaint = Paint().apply { isFilterBitmap = true }
    private val hotPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        style = Paint.Style.STROKE
        strokeWidth = 5f
    }
    private val src = Rect(0, 0, ThermalPacketParser.WIDTH, ThermalPacketParser.HEIGHT)
    private val dst = RectF()
    @Volatile private var hottest = -1

    /** The visible temperature range of the last frame, for the status line. */
    @Volatile var range = 0f to 0f
        private set

    init {
        setBackgroundColor(Color.BLACK)
    }

    /** Safe to call from any thread. */
    fun update(frame: ThermalFrame) {
        val t = frame.celsius
        val sorted = t.sortedArray()
        val low = sorted[(sorted.size * 0.02f).toInt()]
        val high = sorted[(sorted.size * 0.98f).toInt()]
        val span = (high - low).coerceAtLeast(0.5f)
        var hot = 0
        for (i in t.indices) {
            colors[i] = heatColor((t[i] - low) / span)
            if (t[i] > t[hot]) hot = i
        }
        synchronized(bitmap) {
            bitmap.setPixels(colors, 0, ThermalPacketParser.WIDTH, 0, 0, ThermalPacketParser.WIDTH, ThermalPacketParser.HEIGHT)
        }
        hottest = hot
        range = low to high
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        val scale = minOf(width.toFloat() / ThermalPacketParser.WIDTH, height.toFloat() / ThermalPacketParser.HEIGHT)
        val w = ThermalPacketParser.WIDTH * scale
        val h = ThermalPacketParser.HEIGHT * scale
        dst.set((width - w) / 2, (height - h) / 2, (width + w) / 2, (height + h) / 2)
        synchronized(bitmap) { canvas.drawBitmap(bitmap, src, dst, bitmapPaint) }
        val hot = hottest
        if (hot >= 0) {
            val x = dst.left + (hot % ThermalPacketParser.WIDTH + 0.5f) * scale
            val y = dst.top + (hot / ThermalPacketParser.WIDTH + 0.5f) * scale
            canvas.drawCircle(x, y, scale * 0.8f, hotPaint)
        }
    }

    private fun heatColor(value: Float): Int {
        val scaled = value.coerceIn(0f, 0.9999f) * (STOPS.size - 1)
        val i = scaled.toInt()
        val mix = scaled - i
        val a = STOPS[i]
        val b = STOPS[i + 1]
        return Color.rgb(
            (a[0] + (b[0] - a[0]) * mix).toInt(),
            (a[1] + (b[1] - a[1]) * mix).toInt(),
            (a[2] + (b[2] - a[2]) * mix).toInt(),
        )
    }

    companion object {
        // Same palette as test_thermal_stream/public/app.js: cold blue -> green -> yellow -> red -> white.
        private val STOPS = arrayOf(
            intArrayOf(0, 0, 80), intArrayOf(0, 70, 255), intArrayOf(0, 220, 255), intArrayOf(40, 220, 80),
            intArrayOf(255, 235, 0), intArrayOf(255, 40, 0), intArrayOf(255, 255, 255),
        )
    }
}
