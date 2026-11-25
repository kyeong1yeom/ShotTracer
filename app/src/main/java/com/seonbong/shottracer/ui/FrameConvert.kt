package com.seonbong.shottracer.ui

import android.graphics.Bitmap
import androidx.camera.core.ImageProxy
import com.seonbong.shottracer.engine.GrayFrame
import kotlin.math.max

/** 분석용 프레임의 긴 변 길이(px). 클수록 작은 공도 잡지만 느려진다. */
const val ANALYSIS_LONG_SIDE = 640

object FrameConvert {

    private var raw = ByteArray(0)   // 분석 스레드 전용 재사용 버퍼

    /**
     * CameraX YUV 이미지의 Y 채널 → 똑바로 세운(회전 보정) 축소 흑백 프레임.
     * cropRect(미리보기와 같은 화면 영역)만 사용한다.
     * 1) 버퍼를 한 번에 배열로 복사 2) 센서 방향 그대로 축소(연속 메모리 접근) 3) 작은 이미지만 회전
     */
    fun fromImageProxy(image: ImageProxy): GrayFrame {
        val plane = image.planes[0]
        val buf = plane.buffer
        buf.rewind()
        val len = buf.remaining()
        if (raw.size < len) raw = ByteArray(len)
        buf.get(raw, 0, len)
        val rowStride = plane.rowStride
        val ps = plane.pixelStride
        val crop = image.cropRect
        val cw = crop.width()
        val ch = crop.height()
        val step = max(1, (max(cw, ch) + ANALYSIS_LONG_SIDE - 1) / ANALYSIS_LONG_SIDE)
        val sw = cw / step
        val sh = ch / step
        val small = IntArray(sw * sh)
        for (y in 0 until sh * step) {
            val row = (crop.top + y) * rowStride + crop.left * ps
            val oRow = (y / step) * sw
            var idx = row
            for (x in 0 until sw * step) {
                small[oRow + x / step] += raw[idx].toInt() and 0xFF
                idx += ps
            }
        }
        val area = step * step
        val rot = image.imageInfo.rotationDegrees
        val dw = if (rot == 90 || rot == 270) sh else sw
        val dh = if (rot == 90 || rot == 270) sw else sh
        val out = ByteArray(dw * dh)
        for (uy in 0 until dh) for (ux in 0 until dw) {
            // 똑바로 선 좌표(ux,uy) → 축소된 센서 좌표(sx,sy)
            val sx: Int; val sy: Int
            when (rot) {
                90 -> { sx = uy; sy = sh - 1 - ux }
                180 -> { sx = sw - 1 - ux; sy = sh - 1 - uy }
                270 -> { sx = sw - 1 - uy; sy = ux }
                else -> { sx = ux; sy = uy }
            }
            out[uy * dw + ux] = (small[sy * sw + sx] / area).toByte()
        }
        return GrayFrame(dw, dh, out, image.imageInfo.timestamp / 1e9)
    }

    /** 영상에서 꺼낸 비트맵 → 축소 흑백 프레임 */
    fun fromBitmap(src: Bitmap, timeSec: Double): GrayFrame {
        val step = max(1, (max(src.width, src.height) + ANALYSIS_LONG_SIDE - 1) / ANALYSIS_LONG_SIDE)
        val dw = src.width / step
        val dh = src.height / step
        val bmp = if (step == 1) src else Bitmap.createScaledBitmap(src, dw, dh, true)
        val px = IntArray(dw * dh)
        bmp.getPixels(px, 0, dw, 0, 0, dw, dh)
        if (bmp !== src) bmp.recycle()
        val out = ByteArray(dw * dh)
        for (i in px.indices) {
            val c = px[i]
            val y = (((c shr 16) and 0xFF) * 77 + ((c shr 8) and 0xFF) * 150 + (c and 0xFF) * 29) shr 8
            out[i] = y.toByte()
        }
        return GrayFrame(dw, dh, out, timeSec)
    }
}
