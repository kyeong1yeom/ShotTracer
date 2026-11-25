package com.seonbong.shottracer.engine

import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * 골프공 궤적 엔진 v2 (움직임 감지 기반). Python 프로토타입(prototype/engine2.py)과 같은 로직.
 *
 *  1) refineBall   : 손가락 탭 위치 → 실제 공 중심·크기로 보정
 *  2) ImpactDetector: "공 자체가 티에서 사라져 계속 안 보일 때"만 임팩트 (클럽 왜글·백스윙 무시)
 *  3) BallTracker  : 포착 단계 — 티에서 일직선으로 멀어지는 후보 묶음을 공으로 선택(클럽은 호를 그림)
 *                    추적 단계 — 예측 위치 근처 탐색, 잔상(streak) 허용, 비행 곡선으로 이상점 차단
 *  4) TrajectoryFitter: 원근+공기저항 탄도 모델로 나머지 비행 예측
 */

class GrayFrame(val width: Int, val height: Int, val data: ByteArray, val timeSec: Double) {
    fun at(x: Int, y: Int): Int = data[y * width + x].toInt() and 0xFF
}

data class TrackPoint(val t: Double, val x: Double, val y: Double)

class Trajectory(val impactTime: Double, val points: List<TrackPoint>, val tracked: List<TrackPoint>)

/** 공 위치(px), 반지름(px), 밝기 방향(+1: 주변보다 밝음) */
data class BallInfo(val x: Double, val y: Double, val r: Double, val polarity: Int)

// ================================================================ 연결 성분(blob)

internal class Blob(
    val area: Int, val cx: Double, val cy: Double, val meanVal: Double,
    val xs: IntArray, val ys: IntArray, val bw: Int, val bh: Int,
) {
    /** 공분산 고유값 (작은 것, 큰 것)과 큰 축 방향 */
    val evSmall: Double
    val evBig: Double
    val majorX: Double
    val majorY: Double

    init {
        if (area < 4) {
            evSmall = 0.0; evBig = 0.0; majorX = 1.0; majorY = 0.0
        } else {
            var sxx = 0.0; var syy = 0.0; var sxy = 0.0
            for (i in 0 until area) {
                val dx = xs[i] - cx; val dy = ys[i] - cy
                sxx += dx * dx; syy += dy * dy; sxy += dx * dy
            }
            val n1 = (area - 1).toDouble()
            sxx /= n1; syy /= n1; sxy /= n1
            val tr = sxx + syy
            val disc = sqrt(max(0.0, (sxx - syy) * (sxx - syy) / 4 + sxy * sxy))
            evBig = tr / 2 + disc
            evSmall = max(0.0, tr / 2 - disc)
            // 큰 고유값의 고유벡터
            var vx: Double; var vy: Double
            if (abs(sxy) > 1e-9) { vx = evBig - syy; vy = sxy } else if (sxx >= syy) { vx = 1.0; vy = 0.0 } else { vx = 0.0; vy = 1.0 }
            val n = hypot(vx, vy)
            majorX = vx / n; majorY = vy / n
        }
    }

    fun elongation(): Double = evBig / max(evSmall, 0.25)
}

/** 재사용 버퍼로 마스크의 8-연결 성분을 찾는다. values 는 성분 평균값 계산용(선택) */
internal class BlobFinder {
    private var visited = BooleanArray(0)
    private var stack = IntArray(0)
    private var list = IntArray(0)

    fun find(mask: BooleanArray, values: IntArray?, ww: Int, hh: Int, x0: Int, y0: Int): List<Blob> {
        val size = ww * hh
        if (visited.size < size) { visited = BooleanArray(size); stack = IntArray(size); list = IntArray(size) }
        java.util.Arrays.fill(visited, 0, size, false)
        val out = ArrayList<Blob>()
        for (i0 in 0 until size) {
            if (!mask[i0] || visited[i0]) continue
            var sp = 0; var n = 0
            stack[sp++] = i0; visited[i0] = true
            while (sp > 0) {
                val i = stack[--sp]
                list[n++] = i
                val px = i % ww; val py = i / ww
                for (dy in -1..1) for (dx in -1..1) {
                    if (dx == 0 && dy == 0) continue
                    val nx = px + dx; val ny = py + dy
                    if (nx < 0 || ny < 0 || nx >= ww || ny >= hh) continue
                    val j = ny * ww + nx
                    if (mask[j] && !visited[j]) { visited[j] = true; stack[sp++] = j }
                }
            }
            val xs = IntArray(n); val ys = IntArray(n)
            var sx = 0.0; var sy = 0.0; var sv = 0.0
            var minX = Int.MAX_VALUE; var maxX = Int.MIN_VALUE; var minY = Int.MAX_VALUE; var maxY = Int.MIN_VALUE
            for (k in 0 until n) {
                val i = list[k]
                val x = i % ww + x0; val y = i / ww + y0
                xs[k] = x; ys[k] = y; sx += x; sy += y
                if (values != null) sv += values[i]
                minX = min(minX, x); maxX = max(maxX, x); minY = min(minY, y); maxY = max(maxY, y)
            }
            out.add(Blob(n, sx / n, sy / n, sv / n, xs, ys, maxX - minX + 1, maxY - minY + 1))
        }
        return out
    }
}

// ================================================================ 1) 탭 위치 → 공

fun refineBall(f: GrayFrame, tapX: Double, tapY: Double): BallInfo {
    val w = f.width; val h = f.height
    val big = max(10, (0.04 * w).toInt())
    val tx = tapX.roundToInt(); val ty = tapY.roundToInt()
    val x0 = max(0, tx - big); val x1 = min(w, tx + big + 1)
    val y0 = max(0, ty - big); val y1 = min(h, ty + big + 1)
    val fallback = BallInfo(tapX, tapY, max(2.0, w * 0.01), 1)
    if (x1 <= x0 || y1 <= y0) return fallback
    val ww = x1 - x0; val hh = y1 - y0
    val v = IntArray(ww * hh)
    for (y in y0 until y1) for (x in x0 until x1) v[(y - y0) * ww + (x - x0)] = f.at(x, y)
    val med = median(v.map { it.toDouble() }.toDoubleArray())
    val finder = BlobFinder()
    var best: BallInfo? = null; var bestScore = -Double.MAX_VALUE
    val maxArea = (0.04 * w) * (0.04 * w) * 3.2
    for (pol in intArrayOf(1, -1)) {
        val c = IntArray(v.size) { ((v[it] - med) * pol).toInt() }
        val peak = c.maxOrNull() ?: 0
        if (peak < 20) continue
        val mask = BooleanArray(v.size) { c[it] > 0.5 * peak }
        for (b in finder.find(mask, c, ww, hh, x0, y0)) {
            if (b.area > maxArea || max(b.bw, b.bh) > 2.5 * min(b.bw, b.bh) + 2) continue  // 클럽·몸
            val d = hypot(b.cx - tapX, b.cy - tapY)
            val score = b.meanVal - 1.5 * d - (if (pol < 0) 25 else 0)   // 흰 공이 일반적
            if (score > bestScore) {
                bestScore = score
                best = BallInfo(b.cx, b.cy, max(1.5, sqrt(b.area / Math.PI) * 1.1), pol)
            }
        }
    }
    return best ?: fallback
}

// ================================================================ 2) 임팩트 감지

class ImpactDetector(private val ball: BallInfo) {
    private var c0 = Double.NaN
    private var runStart = Double.NaN
    private var runN = 0
    var impactTime = Double.NaN
        private set

    /** 공 중심부 밝기 - 주변 고리 중앙값 (공 방향 기준) */
    private fun contrast(f: GrayFrame): Double {
        val r = ball.r
        val rc = max(1.0, r * 0.7)
        val big = ceil(r * 3.2).toInt() + 1
        val x0 = max(0, (ball.x - big).toInt()); val x1 = min(f.width, (ball.x + big + 1).toInt())
        val y0 = max(0, (ball.y - big).toInt()); val y1 = min(f.height, (ball.y + big + 1).toInt())
        var coreSum = 0.0; var coreN = 0
        val ring = ArrayList<Double>()
        for (y in y0 until y1) for (x in x0 until x1) {
            val d = hypot(x - ball.x, y - ball.y)
            val p = f.at(x, y).toDouble()
            if (d <= rc) { coreSum += p; coreN++ }
            if (d >= r * 1.8 && d <= r * 3.2) ring.add(p)
        }
        if (coreN == 0 || ring.isEmpty()) return 0.0
        return (coreSum / coreN - median(ring.toDoubleArray())) * ball.polarity
    }

    /** 임팩트가 확정되는 프레임에서 한 번 true (impactTime = 공이 처음 사라진 시각) */
    fun update(f: GrayFrame): Boolean {
        val c = contrast(f)
        if (c0.isNaN()) { c0 = max(c, 12.0); return false }
        val absent = c < 0.45 * c0
        if (!absent) {
            runStart = Double.NaN; runN = 0
            c0 = 0.97 * c0 + 0.03 * max(c, 12.0)   // 조명 변화 따라가기
            return false
        }
        if (runStart.isNaN()) runStart = f.timeSec
        runN++
        if (runN >= 2 && f.timeSec - runStart >= 0.025) { impactTime = runStart; return true }
        return false
    }

    fun ballPresent(f: GrayFrame): Boolean = !c0.isNaN() && contrast(f) >= 0.6 * c0
}

// ================================================================ 3) 공 추적

class BallTracker(
    private val startX: Double, private val startY: Double, private val ballR: Double,
    private val w: Int, private val h: Int, private val polarity: Int = 1, private val diffThr: Int = 14,
) {
    companion object {
        const val MAX_GAP_SEC = 0.12
        const val MAX_BUF = 61
        const val MAX_MISS_SEC = 0.2
        const val ACQUIRE_SEC = 0.35
    }

    private class Cand(val t: Double, val x: Double, val y: Double, val ang: Double, val dist: Double)

    private val frames = ArrayDeque<GrayFrame>()
    val pts = ArrayList<TrackPoint>()
    private val cands = ArrayList<Cand>()
    var acquired = false
        private set
    var done = false
        private set
    private var tFirst = Double.NaN
    private var miss = 0
    private var model: FlightModel? = null
    private val finder = BlobFinder()
    private var maskBuf = BooleanArray(0)
    private var dbBuf = IntArray(0)

    // ---------- 운동 모델
    private fun velocity(): DoubleArray {
        val n = min(4, pts.size)
        if (n < 2) return doubleArrayOf(0.0, 0.0)
        val q = pts.subList(pts.size - n, pts.size)
        val tl = q.last().t
        if (tl - q.first().t < 1e-6) return doubleArrayOf(0.0, 0.0)
        val tt = DoubleArray(n) { q[it].t - tl }
        return doubleArrayOf(linearSlope(tt, DoubleArray(n) { q[it].x }), linearSlope(tt, DoubleArray(n) { q[it].y }))
    }

    private fun speed(): Double { val v = velocity(); return hypot(v[0], v[1]) }

    private fun frameDt(): Double =
        if (frames.size < 2) 1.0 / 30 else max(frames.last().timeSec - frames[frames.size - 2].timeSec, 1e-4)

    private fun maxGap(): Int =
        if (frames.size < 2) 1 else max(1, min((MAX_BUF - 1) / 2, (MAX_GAP_SEC / frameDt()).roundToInt()))

    /** 멀리서 느려진 공도 잡도록 차분 간격(프레임 수)을 늘린다. 잔상끼리 겹치지 않게 넉넉히 */
    private fun gap(): Int {
        if (!acquired || pts.size < 2 || frames.size < 2) return 1
        val v = speed() * frameDt()
        val need = 4 * ballR + 6
        return min(maxGap(), max(1, ceil(need / max(v, 0.05)).toInt()))
    }

    // ---------- blob 도우미
    private fun blobs(a: GrayFrame, b: GrayFrame, c: GrayFrame, x0: Int, x1: Int, y0: Int, y1: Int): List<Blob> {
        val ww = x1 - x0; val hh = y1 - y0; val size = ww * hh
        if (maskBuf.size < size) { maskBuf = BooleanArray(size); dbBuf = IntArray(size) }
        for (y in y0 until y1) {
            val row = (y - y0) * ww
            for (x in x0 until x1) {
                val vb = b.at(x, y)
                val d1 = vb - a.at(x, y)
                val d2 = c.at(x, y) - vb
                dbBuf[row + x - x0] = d1
                maskBuf[row + x - x0] = abs(d1) > diffThr && abs(d2) > diffThr
            }
        }
        return finder.find(maskBuf, dbBuf, ww, hh, x0, y0)
    }

    private fun sizeOk(b: Blob): Boolean {
        if (b.area <= max(30.0, (ballR * 4) * (ballR * 4))) return true
        val thick = 4 * sqrt(b.evSmall)
        return thick <= 2.5 * ballR + 3 && b.evBig >= 9 * max(b.evSmall, 0.25)   // 가는 잔상 = 흐려진 공
    }

    /** 잔상이면 진행 방향 앞쪽 끝이 "지금" 공의 위치 */
    private fun lead(b: Blob, mx: Double, my: Double): DoubleArray {
        if (b.area < 4 || b.elongation() < 9) return doubleArrayOf(b.cx, b.cy)
        var ax = b.majorX; var ay = b.majorY
        if (ax * mx + ay * my < 0) { ax = -ax; ay = -ay }
        var pmax = -Double.MAX_VALUE
        for (i in 0 until b.area) pmax = max(pmax, ax * (b.xs[i] - b.cx) + ay * (b.ys[i] - b.cy))
        val l = max(0.0, pmax - ballR)
        return doubleArrayOf(b.cx + ax * l, b.cy + ay * l)
    }

    /** 클럽이 닿는 범위: 여기서 공보다 어두운 움직임은 클럽/몸 */
    private fun clubZone(x: Double, y: Double) = hypot(x - startX, y - startY) < 0.35 * h

    // ---------- 메인
    fun push(f: GrayFrame) {
        if (done) return
        if (tFirst.isNaN()) tFirst = f.timeSec
        frames.addLast(f)
        while (frames.size > MAX_BUF) frames.removeFirst()
        if (!acquired) {
            if (frames.size >= 3) acquireStep()
            if (!acquired && f.timeSec - tFirst > ACQUIRE_SEC) done = true
            return
        }
        val g = gap()
        if (frames.size < 2 * g + 1) return
        val a = frames[frames.size - 1 - 2 * g]; val b = frames[frames.size - 1 - g]; val c = frames.last()
        val tb = b.timeSec
        if (tb <= pts.last().t) return
        val last = pts.last()
        val v = velocity()
        val dt = tb - last.t
        val fullX = last.x + v[0] * dt; val fullY = last.y + v[1] * dt
        val px = last.x + 0.75 * (fullX - last.x); val py = last.y + 0.75 * (fullY - last.y)
        val step = hypot(fullX - last.x, fullY - last.y)
        val base = max(8.0, step * 0.45 + 6)
        val rad = min(base + miss, 2 * base)
        val x0 = max(0, (px - rad).toInt()); val x1 = min(w, (px + rad + 1).toInt())
        val y0 = max(0, (py - rad).toInt()); val y1 = min(h, (py + rad + 1).toInt())
        if (x1 <= x0 || y1 <= y0) { onMiss(); return }

        var best: DoubleArray? = null; var bestD = Double.MAX_VALUE
        for (bl in blobs(a, b, c, x0, x1, y0, y1)) {
            if (!sizeOk(bl)) continue
            if (clubZone(bl.cx, bl.cy) && bl.meanVal * polarity <= 0) continue
            val d = hypot(bl.cx - px, bl.cy - py)
            if (d > rad || d >= bestD) continue
            best = lead(bl, v[0], v[1]); bestD = d
        }
        val m = model
        if (best != null && m != null) {
            val p = m.at(tb)
            if (p == null || hypot(p[0] - best[0], p[1] - best[1]) > max(10.0, rad) * (1 + 0.5 * miss)) best = null
        }
        if (best == null) { onMiss(); return }
        pts.add(TrackPoint(tb, best[0], best[1])); miss = 0
        // 비행이 충분히 길어지면 곡선으로 이상점(노이즈·다른 물체)을 막는다
        if (pts.size >= 10 && pts.size % 5 == 0 && pts.last().t - pts.first().t >= 0.3) {
            val good = TrajectoryFitter.robust(startX, startY, pts)
            model = if (good.size >= 8) TrajectoryFitter.fitModel(startX, startY, good) else null
        }
    }

    private fun acquireStep() {
        val a = frames[frames.size - 3]; val b = frames[frames.size - 2]; val c = frames.last()
        val tb = b.timeSec
        val x0 = max(0, (startX - 0.35 * w).toInt()); val x1 = min(w, (startX + 0.35 * w + 1).toInt())
        val y0 = max(0, (startY - 0.7 * h).toInt()); val y1 = min(h, (startY - ballR * 1.5).toInt())
        if (x1 <= x0 || y1 <= y0) return
        for (bl in blobs(a, b, c, x0, x1, y0, y1)) {
            if (bl.meanVal * polarity <= 0) continue
            if (!sizeOk(bl)) continue
            val rx = bl.cx - startX; val ry = bl.cy - startY
            val nr = hypot(rx, ry)
            if (nr < 1) continue
            if (bl.elongation() >= 4 && abs((bl.majorX * rx + bl.majorY * ry) / nr) < cos(Math.toRadians(25.0))) continue
            val lp = lead(bl, rx, ry)
            cands.add(Cand(tb, lp[0], lp[1], Math.toDegrees(atan2(lp[1] - startY, lp[0] - startX)), hypot(lp[0] - startX, lp[1] - startY)))
        }
        tryChain()
    }

    /** 같은 방향(±4°)에 있고 시간이 갈수록 티에서 멀어지는 후보 묶음 = 공 */
    private fun tryChain() {
        if (cands.size < 3) return
        val need = if (frameDt() < 1.0 / 50) 4 else 3
        var bestChain: List<Cand>? = null; var bestScore = -Double.MAX_VALUE
        for (seed in cands) {
            val group = HashMap<Long, Cand>()
            for (cd in cands) {
                val da = abs(cd.ang - seed.ang)
                if (da >= 4.0) continue
                val key = Math.round(cd.t * 1e6)
                val cur = group[key]
                if (cur == null || abs(cur.ang - seed.ang) > da) group[key] = cd
            }
            if (group.size < need) continue
            val seq = group.values.sortedBy { it.t }
            val len = IntArray(seq.size) { 1 }; val prev = IntArray(seq.size) { -1 }
            for (i in seq.indices) for (j in 0 until i) {
                if (seq[j].dist + 0.5 < seq[i].dist && len[j] + 1 > len[i]) { len[i] = len[j] + 1; prev[i] = j }
            }
            var i = len.indices.maxByOrNull { len[it] } ?: continue
            if (len[i] < need) continue
            val chain = ArrayList<Cand>()
            while (i >= 0) { chain.add(seq[i]); i = prev[i] }
            chain.reverse()
            val mean = chain.map { it.ang }.average()
            val spread = sqrt(chain.map { (it.ang - mean) * (it.ang - mean) }.average())
            val score = chain.size - 0.2 * spread
            if (score > bestScore) { bestScore = score; bestChain = chain }
        }
        val bc = bestChain ?: return
        pts.clear()
        for (cd in bc) pts.add(TrackPoint(cd.t, cd.x, cd.y))
        acquired = true
        cands.clear()
    }

    private fun onMiss() {
        miss++
        if (frames.last().timeSec - pts.last().t > MAX_MISS_SEC + MAX_GAP_SEC) done = true
    }
}

// ================================================================ 4) 궤적 피팅

/** 임팩트 시각 t0 기준 탄도 모델. at(t): 영상 시각 t 의 화면 위치 */
class FlightModel(val t0: Double, private val fn: (Double) -> DoubleArray) {
    fun raw(tt: Double): DoubleArray = fn(tt)
    fun at(t: Double): DoubleArray? {
        val m = fn(t - t0)
        if (m[2] <= 0.05) return null
        return doubleArrayOf(m[0] / m[2], m[1] / m[2])
    }
}

object TrajectoryFitter {

    /**
     * 원근 투영 + 공기저항 탄도 모델
     *   u(t) = (A0 + A1·φ + A2·t²) / (1 + a·φ),  v(t) = (B0 + B1·φ + B2·t²) / (1 + a·φ),  φ = (1 - e^{-kt}) / k
     */
    fun fitModel(startX: Double, startY: Double, pts: List<TrackPoint>): FlightModel? {
        if (pts.size < 3) return null
        val k3 = min(2, pts.size - 1)
        val step = hypot(pts[k3].x - pts[0].x, pts[k3].y - pts[0].y) / max(1e-6, pts[k3].t - pts[0].t)
        val back = hypot(pts[0].x - startX, pts[0].y - startY) / max(step, 1e-6)
        val t0 = pts[0].t - back.coerceIn(0.0, 0.25)
        val n = pts.size + 1
        val t = DoubleArray(n); val u = DoubleArray(n); val v = DoubleArray(n); val wt = DoubleArray(n) { 1.0 }
        u[0] = startX; v[0] = startY; wt[0] = 4.0
        for (i in pts.indices) { t[i + 1] = pts[i].t - t0; u[i + 1] = pts[i].x; v[i + 1] = pts[i].y }

        var model: ((Double) -> DoubleArray)? = null
        if (n >= 6) {
            var bestRes = Double.MAX_VALUE
            val ks = DoubleArray(31) { if (it == 0) 0.0 else 0.05 * Math.pow(4.0 / 0.05, (it - 1) / 29.0) }
            for (k in ks) {
                val ph = DoubleArray(n) { phi(t[it], k) }
                val rows = ArrayList<DoubleArray>(2 * n); val rhs = ArrayList<Double>(2 * n); val rw = ArrayList<Double>(2 * n)
                for (i in 0 until n) {
                    rows.add(doubleArrayOf(1.0, ph[i], t[i] * t[i], 0.0, 0.0, 0.0, -ph[i] * u[i])); rhs.add(u[i]); rw.add(wt[i])
                    rows.add(doubleArrayOf(0.0, 0.0, 0.0, 1.0, ph[i], t[i] * t[i], -ph[i] * v[i])); rhs.add(v[i]); rw.add(wt[i])
                }
                val s = weightedLeastSquares(rows, rhs, rw) ?: continue
                var ok = true; var res = 0.0
                for (i in 0 until n) {
                    val d = 1 + s[6] * ph[i]
                    if (d <= 0.05) { ok = false; break }
                    val eu = (s[0] + s[1] * ph[i] + s[2] * t[i] * t[i]) / d - u[i]
                    val ev = (s[3] + s[4] * ph[i] + s[5] * t[i] * t[i]) / d - v[i]
                    res += wt[i] * (eu * eu + ev * ev)
                }
                if (ok && res < bestRes) {
                    bestRes = res
                    val kk = k
                    model = { tt: Double ->
                        val p = phi(tt, kk)
                        doubleArrayOf(s[0] + s[1] * p + s[2] * tt * tt, s[3] + s[4] * p + s[5] * tt * tt, 1 + s[6] * p)
                    }
                }
            }
        }
        if (model == null) {
            val cu = polyfit(t, u, 2, wt) ?: return null
            val cv = polyfit(t, v, 2, wt) ?: return null
            model = { tt: Double -> doubleArrayOf(polyval(cu, tt), polyval(cv, tt), 1.0) }
        }
        val fn: (Double) -> DoubleArray = model ?: return null
        return FlightModel(t0, fn)
    }

    /** 곡선에서 확실히 벗어난 점을 하나씩 빼며 다시 맞춘다 */
    fun robust(startX: Double, startY: Double, input: List<TrackPoint>): List<TrackPoint> {
        val p = ArrayList(input)
        repeat(max(0, p.size / 3)) {
            if (p.size < 6) return p
            val m = fitModel(startX, startY, p) ?: return p
            val res = DoubleArray(p.size) { i ->
                val q = m.at(p[i].t)
                if (q == null) 1e9 else hypot(q[0] - p[i].x, q[1] - p[i].y)
            }
            var k = 0
            for (i in res.indices) if (res[i] > res[k]) k = i
            if (res[k] <= max(6.0, 4.0 * median(res))) return p
            p.removeAt(k)
        }
        return p
    }

    fun fit(
        startX: Double, startY: Double, raw: List<TrackPoint>, w: Int, h: Int,
        landingFraction: Double = 0.3, tMax: Double = 8.0, dt: Double = 1.0 / 120,
    ): Trajectory? {
        if (raw.size < 3) return null
        var pts = removeOutliers(raw)
        if (pts.size >= 6) pts = robust(startX, startY, pts)
        if (pts.size < 3) return null
        val m = fitModel(startX, startY, pts) ?: return null

        val out = ArrayList<TrackPoint>()
        var apexV = Double.NaN; var apexSeen = false
        var tt = 0.0
        while (tt <= tMax) {
            val r = m.raw(tt)
            if (r[2] < 0.05) break
            val x = r[0] / r[2]; val y = r[1] / r[2]
            if (apexV.isNaN() || y < apexV) apexV = y
            else if (y > apexV + 1) apexSeen = true
            if (apexSeen && y >= apexV + landingFraction * (startY - apexV)) { out.add(TrackPoint(m.t0 + tt, x, y)); break }
            if (x < -w || x > 2.0 * w || y < -h || y > 2.0 * h) break
            out.add(TrackPoint(m.t0 + tt, x, y))
            tt += dt
        }
        return Trajectory(m.t0, out, pts)
    }

    private fun phi(t: Double, k: Double) = if (k < 1e-6) t else (1.0 - exp(-k * t)) / k

    /** 주변 점들로 만든 국소 2차식에서 6px 이상 벗어난 검출점 제거 */
    private fun removeOutliers(p: List<TrackPoint>): List<TrackPoint> {
        if (p.size < 7) return p
        val keep = BooleanArray(p.size) { true }
        for (i in p.indices) {
            val idx = (max(0, i - 4) until min(p.size, i + 5)).filter { it != i }
            if (idx.size < 4) continue
            val tt = DoubleArray(idx.size) { p[idx[it]].t - p[i].t }
            val cx = polyfit(tt, DoubleArray(idx.size) { p[idx[it]].x }, 2, null) ?: continue
            val cy = polyfit(tt, DoubleArray(idx.size) { p[idx[it]].y }, 2, null) ?: continue
            if (hypot(polyval(cx, 0.0) - p[i].x, polyval(cy, 0.0) - p[i].y) > 6) keep[i] = false
        }
        return p.filterIndexed { i, _ -> keep[i] }
    }
}

// ================================================================ 수치 계산 도우미

internal fun median(a: DoubleArray): Double {
    if (a.isEmpty()) return 0.0
    val s = a.sortedArray()
    val m = s.size / 2
    return if (s.size % 2 == 1) s[m] else (s[m - 1] + s[m]) / 2
}

internal fun linearSlope(t: DoubleArray, y: DoubleArray): Double {
    val mt = t.average(); val my = y.average()
    var num = 0.0; var den = 0.0
    for (i in t.indices) { num += (t[i] - mt) * (y[i] - my); den += (t[i] - mt) * (t[i] - mt) }
    return if (den < 1e-12) 0.0 else num / den
}

/** 계수 오름차순 [c0, c1, c2] */
internal fun polyfit(t: DoubleArray, y: DoubleArray, deg: Int, w: DoubleArray?): DoubleArray? {
    val rows = ArrayList<DoubleArray>(); val rhs = ArrayList<Double>(); val rw = ArrayList<Double>()
    for (i in t.indices) {
        rows.add(DoubleArray(deg + 1) { p -> Math.pow(t[i], p.toDouble()) }); rhs.add(y[i]); rw.add(w?.get(i) ?: 1.0)
    }
    return weightedLeastSquares(rows, rhs, rw)
}

internal fun polyval(c: DoubleArray, t: Double): Double {
    var r = 0.0
    for (p in c.indices.reversed()) r = r * t + c[p]
    return r
}

/** 열 정규화 + 정규방정식 + 부분 피벗 가우스 소거 */
internal fun weightedLeastSquares(rows: List<DoubleArray>, rhs: List<Double>, w: List<Double>): DoubleArray? {
    if (rows.isEmpty()) return null
    val m = rows[0].size
    val scale = DoubleArray(m)
    for (r in rows.indices) for (j in 0 until m) scale[j] += (rows[r][j] * w[r]) * (rows[r][j] * w[r])
    for (j in 0 until m) scale[j] = if (scale[j] > 0) sqrt(scale[j]) else 1.0
    val ata = Array(m) { DoubleArray(m) }
    val atb = DoubleArray(m)
    for (r in rows.indices) {
        val ww = w[r] * w[r]
        for (i in 0 until m) {
            val ai = rows[r][i] / scale[i]
            atb[i] += ww * ai * rhs[r]
            for (j in 0 until m) ata[i][j] += ww * ai * rows[r][j] / scale[j]
        }
    }
    val x = solve(ata, atb) ?: return null
    for (j in 0 until m) x[j] /= scale[j]
    return x
}

private fun solve(a: Array<DoubleArray>, b: DoubleArray): DoubleArray? {
    val n = b.size
    for (c in 0 until n) {
        var p = c
        for (r in c + 1 until n) if (abs(a[r][c]) > abs(a[p][c])) p = r
        if (abs(a[p][c]) < 1e-12) return null
        val tmp = a[c]; a[c] = a[p]; a[p] = tmp
        val tb = b[c]; b[c] = b[p]; b[p] = tb
        for (r in c + 1 until n) {
            val f = a[r][c] / a[c][c]
            for (k in c until n) a[r][k] -= f * a[c][k]
            b[r] -= f * b[c]
        }
    }
    val x = DoubleArray(n)
    for (r in n - 1 downTo 0) {
        var s = b[r]
        for (k in r + 1 until n) s -= a[r][k] * x[k]
        x[r] = s / a[r][r]
    }
    return x
}
