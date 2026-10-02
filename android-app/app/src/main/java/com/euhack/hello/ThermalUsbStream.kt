package com.euhack.hello

import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.hardware.usb.UsbConstants
import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbDeviceConnection
import android.hardware.usb.UsbEndpoint
import android.hardware.usb.UsbInterface
import android.hardware.usb.UsbManager
import android.os.Build
import android.os.SystemClock
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.zip.CRC32

/** One calibrated MLX90640 image: 32x24 temperatures in degrees C, row-major. */
class ThermalFrame(
    val sequence: Long,
    val subpage: Int,
    val sensorTimeUs: Long,
    /** SystemClock.elapsedRealtimeNanos() when the packet was parsed. */
    val receivedNs: Long,
    val celsius: FloatArray,
    val ambientC: Float,
    /** The THM2 packet exactly as received, for session recordings. */
    val packet: ByteArray,
)

/**
 * Reads the QT Py thermal stream over USB OTG. The phone powers the board through the same cable.
 *
 * The ESP32-S3's built-in USB-Serial-JTAG is a CDC-ACM device; only its bulk IN endpoint is used.
 * No control-line requests are sent: toggling DTR/RTS on that port can reset the chip.
 * The device is polled for, so plugging and unplugging the board while running just works.
 */
class ThermalUsbStream(
    context: Context,
    private val onFrame: (ThermalFrame) -> Unit,
    private val onStatus: (String) -> Unit,
) {
    private val appContext = context.applicationContext
    private val usb = appContext.getSystemService(Context.USB_SERVICE) as UsbManager
    @Volatile private var running = false
    private var thread: Thread? = null
    private var permissionAsked: String? = null

    fun start() {
        if (running) return
        running = true
        thread = Thread(::run, "thermal-usb").apply { start() }
    }

    fun stop() {
        running = false
        thread?.join(1000)
        thread = null
    }

    private fun run() {
        val parser = ThermalPacketParser(onFrame)
        var lastStatus = ""
        fun status(text: String) {
            if (text != lastStatus) onStatus(text)
            lastStatus = text
        }
        while (running) {
            val device = usb.deviceList.values.firstOrNull { it.vendorId == ESPRESSIF_VID }
            if (device == null) {
                permissionAsked = null
                status("Plug the thermal camera into the phone (USB-C OTG)")
                SystemClock.sleep(SCAN_INTERVAL_MS)
                continue
            }
            if (!usb.hasPermission(device)) {
                if (permissionAsked != device.deviceName) {
                    permissionAsked = device.deviceName
                    requestPermission(device)
                }
                status("Allow USB access to the thermal camera")
                SystemClock.sleep(SCAN_INTERVAL_MS)
                continue
            }
            val port = openDataPort(device)
            if (port == null) {
                status("Thermal camera found but its serial port could not be opened")
                SystemClock.sleep(SCAN_INTERVAL_MS)
                continue
            }
            status("Thermal camera connected — waiting for frames")
            val (connection, iface, endpoint) = port
            val buffer = ByteArray(READ_BYTES)
            try {
                while (running) {
                    val n = connection.bulkTransfer(endpoint, buffer, buffer.size, READ_TIMEOUT_MS)
                    // A timeout also returns -1; only treat it as an unplug once the device is gone.
                    if (n < 0 && usb.deviceList.values.none { it.deviceName == device.deviceName }) break
                    if (n > 0) parser.push(buffer, n)
                }
            } finally {
                connection.releaseInterface(iface)
                connection.close()
            }
            parser.reset()
        }
    }

    private fun requestPermission(device: UsbDevice) {
        // The answer is not listened for: the scan loop sees hasPermission() flip on its next pass.
        val flags = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) PendingIntent.FLAG_MUTABLE else 0
        val intent = Intent(ACTION_USB_PERMISSION).setPackage(appContext.packageName)
        usb.requestPermission(device, PendingIntent.getBroadcast(appContext, 0, intent, flags))
    }

    /** The CDC data interface (class 0x0A) and its bulk IN endpoint. */
    private fun openDataPort(device: UsbDevice): Triple<UsbDeviceConnection, UsbInterface, UsbEndpoint>? {
        for (i in 0 until device.interfaceCount) {
            val iface = device.getInterface(i)
            if (iface.interfaceClass != UsbConstants.USB_CLASS_CDC_DATA) continue
            val endpoint = (0 until iface.endpointCount).map { iface.getEndpoint(it) }.firstOrNull {
                it.type == UsbConstants.USB_ENDPOINT_XFER_BULK && it.direction == UsbConstants.USB_DIR_IN
            } ?: continue
            val connection = usb.openDevice(device) ?: return null
            if (!connection.claimInterface(iface, true)) {
                connection.close()
                return null
            }
            return Triple(connection, iface, endpoint)
        }
        Log.w(TAG, "no CDC data interface on ${device.deviceName}")
        return null
    }

    companion object {
        private const val TAG = "ThermalUsb"
        private const val ACTION_USB_PERMISSION = "com.euhack.hello.USB_PERMISSION"
        const val ESPRESSIF_VID = 0x303A
        private const val SCAN_INTERVAL_MS = 1000L
        private const val READ_TIMEOUT_MS = 250
        private const val READ_BYTES = 16 * 1024
    }
}

/**
 * Splits the byte stream into THM2 packets (see firmware/components/thermal_stream):
 * 28-byte header, then 768 pixel temperatures and the ambient temperature as int16 centi-degrees C.
 * Resynchronises on the magic and drops anything whose CRC32 does not match.
 */
class ThermalPacketParser(private val onFrame: (ThermalFrame) -> Unit) {
    private val buffer = ByteArray(PACKET_BYTES * 3)
    private var used = 0

    fun reset() {
        used = 0
    }

    fun push(chunk: ByteArray, length: Int) {
        var offset = 0
        while (offset < length) {
            val n = minOf(length - offset, buffer.size - used)
            System.arraycopy(chunk, offset, buffer, used, n)
            used += n
            offset += n
            drain()
        }
    }

    private fun drain() {
        while (used >= MAGIC.size) {
            val start = findMagic()
            if (start < 0) {
                discard(used - (MAGIC.size - 1))
                return
            }
            discard(start)
            if (used < PACKET_BYTES) return
            val frame = decode()
            if (frame == null) {
                discard(1)
                continue
            }
            onFrame(frame)
            discard(PACKET_BYTES)
        }
    }

    private fun findMagic(): Int {
        outer@ for (i in 0..used - MAGIC.size) {
            for (j in MAGIC.indices) if (buffer[i + j] != MAGIC[j]) continue@outer
            return i
        }
        return -1
    }

    private fun discard(count: Int) {
        if (count <= 0) return
        System.arraycopy(buffer, count, buffer, 0, used - count)
        used -= count
    }

    private fun decode(): ThermalFrame? {
        val b = ByteBuffer.wrap(buffer, 0, PACKET_BYTES).order(ByteOrder.LITTLE_ENDIAN)
        val version = b.get(4).toInt() and 0xFF
        val subpage = b.get(5).toInt() and 0xFF
        if (version != VERSION || subpage > 1 || (b.getShort(6).toInt() and 0xFFFF) != HEADER_BYTES ||
            (b.getShort(20).toInt() and 0xFFFF) != PAYLOAD_WORDS ||
            (b.get(22).toInt() and 0xFF) != WIDTH || (b.get(23).toInt() and 0xFF) != HEIGHT
        ) return null
        val crc = CRC32().apply { update(buffer, HEADER_BYTES, PAYLOAD_WORDS * 2) }.value
        if (crc != (b.getInt(24).toLong() and 0xFFFFFFFFL)) return null

        val celsius = FloatArray(PIXELS) { b.getShort(HEADER_BYTES + it * 2) / 100f }
        return ThermalFrame(
            sequence = b.getInt(8).toLong() and 0xFFFFFFFFL,
            subpage = subpage,
            sensorTimeUs = b.getLong(12),
            receivedNs = SystemClock.elapsedRealtimeNanos(),
            celsius = celsius,
            ambientC = b.getShort(HEADER_BYTES + PIXELS * 2) / 100f,
            packet = buffer.copyOf(PACKET_BYTES),
        )
    }

    companion object {
        val MAGIC = byteArrayOf('T'.code.toByte(), 'H'.code.toByte(), 'M'.code.toByte(), '2'.code.toByte())
        const val VERSION = 2
        const val WIDTH = 32
        const val HEIGHT = 24
        const val PIXELS = WIDTH * HEIGHT
        const val HEADER_BYTES = 28
        const val PAYLOAD_WORDS = PIXELS + 1
        const val PACKET_BYTES = HEADER_BYTES + PAYLOAD_WORDS * 2
    }
}
