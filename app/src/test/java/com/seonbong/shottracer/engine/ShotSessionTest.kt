package com.seonbong.shottracer.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.sin

/** 폰 없이 PC에서 엔진만 검사: ./gradlew test */
class ShotSessionTest {

    private val w = 360
    private val h = 640
    private val teeX = 180.0
    private val teeY = 520.0
    private val fps = 60.0

    private fun disc(data: ByteArray, cx: Double, cy: Double, r: Double, v: Int) {
        val ri = r.toInt() + 1
        for (dy in -ri..ri) for (dx in -ri..ri) {
            if (hypot(dx.toDouble(), dy.toDouble()) <= r) {
                val px = (cx + dx).toInt(); val py = (cy + dy).toInt()
                if (px in 0 until w && py in 0 until h) data[py * w + px] = v.toByte()
            }
        }
    }

    /**
     * impactFrame 전: 공은 티 위, 클럽헤드(어두운 원)가 공 바로 옆에서 왜글/백스윙.
     * impactFrame 후: 공은 위로 날아가고(정점 y=260), 클럽헤드는 호를 그리며 위로 지나간다.
     */
    private fun frame(i: Int, impactFrame: Int, withClub: Boolean = true): GrayFrame {
        val data = ByteArray(w * h) { 80.toByte() }
        val k = i - impactFrame
        val t = k / fps
        val bx = if (k < 0) teeX else teeX + 8.0 * t
        val by = if (k < 0) teeY else teeY - 520.0 * t + 260.0 * t * t
        disc(data, bx, by, 3.2, 245)
        if (withClub) {
            val hx = teeX - 60; val hy = teeY - 150
            val rad = hypot(teeX - hx, teeY - hy) - 9
            val a0 = Math.atan2(teeY - hy, teeX - hx)
            val ang = when {
                k < -40 -> a0 + 0.04 * sin(i * 0.6)                       // 왜글
                k < 0 -> a0 + Math.toRadians(150.0) * sin(Math.PI * (k + 40) / 40.0).coerceAtLeast(0.0) // 백·다운스윙
                else -> a0 - Math.toRadians(160.0) * (k / 20.0).coerceAtMost(1.0) // 팔로스루: 위로
            }
            disc(data, hx + rad * cos(ang), hy + rad * sin(ang), 5.0, 35)
        }
        return GrayFrame(w, h, data, i / fps)
    }

    private fun run(impactFrame: Int, total: Int, tapDx: Double = 0.0, tapDy: Double = 0.0, club: Boolean = true): ShotSession {
        val first = frame(0, impactFrame, club)
        val s = ShotSession(first, teeX + tapDx, teeY + tapDy)
        for (i in 0 until total) {
            s.feed(frame(i, impactFrame, club))
            if (s.state == ShotSession.State.FINISHED || s.state == ShotSession.State.FAILED) break
        }
        s.finish()
        return s
    }

    @Test
    fun syntheticShotProducesTrajectory() {
        val s = run(impactFrame = 60, total = 140, club = false)
        assertEquals(ShotSession.State.FINISHED, s.state)
        val tr = s.trajectory
        assertNotNull(tr)
        assertTrue(tr!!.points.size >= 10)
        val apex = tr.points.minOf { it.y }
        assertTrue("정점 오차가 큼: $apex", abs(apex - 260.0) < 8.0)
    }

    @Test
    fun clubNextToBallDoesNotTriggerAndBallIsTracked() {
        // 탭이 공에서 몇 px 빗나가도 보정되어야 함
        val s = run(impactFrame = 60, total = 140, tapDx = 5.0, tapDy = -4.0)
        assertTrue("탭 보정 실패: ${s.ball}", hypot(s.ball.x - teeX, s.ball.y - teeY) < 2.0)
        assertEquals(ShotSession.State.FINISHED, s.state)
        val tr = s.trajectory!!
        assertTrue("임팩트 시각 오차: ${tr.impactTime}", abs(tr.impactTime - 1.0) < 0.06)
        // 추적한 점이 클럽이 아니라 공 위에 있어야 함
        for (p in tr.tracked.take(5)) {
            val k = (p.t * fps) - 60
            val gy = teeY - 520.0 * (k / fps) + 260.0 * (k / fps) * (k / fps)
            assertTrue("공이 아닌 점 추적: $p (공 y=$gy)", abs(p.y - gy) < 12.0)
        }
        val apex = tr.points.minOf { it.y }
        assertTrue("정점 오차가 큼: $apex", abs(apex - 260.0) < 10.0)
    }

    @Test
    fun waggleOnlyNeverTriggersImpact() {
        val s = ShotSession(frame(0, 1000), teeX, teeY)
        for (i in 0 until 50) s.feed(frame(i, 1000))   // 왜글 구간만
        assertEquals(ShotSession.State.ARMED, s.state)
    }
}
