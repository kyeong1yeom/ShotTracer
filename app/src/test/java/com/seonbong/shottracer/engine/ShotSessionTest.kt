package com.seonbong.shottracer.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs
import kotlin.math.hypot

/** 폰 없이 PC에서 엔진만 검사: ./gradlew test */
class ShotSessionTest {

    private val w = 360
    private val h = 640
    private val startX = 180.0
    private val startY = 520.0

    private fun frame(i: Int, impactFrame: Int): GrayFrame {
        val data = ByteArray(w * h) { 80.toByte() }
        val k = i - impactFrame
        val t = k / 60.0
        val bx = if (k < 0) startX else startX + 8.0 * t
        val by = if (k < 0) startY else startY - 520.0 * t + 260.0 * t * t   // 정점 y = 260
        for (dy in -3..3) for (dx in -3..3) {
            if (hypot(dx.toDouble(), dy.toDouble()) <= 3.2) {
                val px = (bx + dx).toInt(); val py = (by + dy).toInt()
                if (px in 0 until w && py in 0 until h) data[py * w + px] = 245.toByte()
            }
        }
        return GrayFrame(w, h, data, i / 60.0)
    }

    @Test
    fun syntheticShotProducesTrajectory() {
        val session = ShotSession(startX, startY, w)
        for (i in 0 until 60) session.feed(frame(i, impactFrame = 5))
        session.finish(w, h)

        assertEquals(ShotSession.State.FINISHED, session.state)
        val tr = session.trajectory
        assertNotNull(tr)
        assertTrue(tr!!.points.size >= 10)
        val apex = tr.points.minOf { it.y }
        assertTrue("정점 오차가 큼: $apex", abs(apex - 260.0) < 5.0)
    }

    @Test
    fun staticBallNeverTriggersImpact() {
        val session = ShotSession(startX, startY, w)
        for (i in 0 until 60) session.feed(frame(i, impactFrame = 1000))
        assertEquals(ShotSession.State.ARMED, session.state)
    }
}
