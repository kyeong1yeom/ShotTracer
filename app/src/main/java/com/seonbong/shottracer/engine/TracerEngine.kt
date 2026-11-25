package com.seonbong.shottracer.engine

import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.exp
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * 골프공 궤적 엔진 (움직임 감지 기반).
 * - 카메라는 삼각대 등으로 고정되어 있다고 가정
 * - 좌표는 모두 분석용 축소 프레임(GrayFrame)의 픽셀 좌표
 * Python 프로토타입(proto/engine.py)과 동일한 로직.
 */

/** 8비트 흑백 프레임 (Y 채널). timeSec = 영상 기준 시간(초) */
class GrayFrame(val width: Int, val height: Int, val data: ByteArray, val timeSec: Double) {
    fun at(x: Int, y: Int): Int = data[y * width + x].toInt() and 0xFF
}

data class TrackPoint(val t: Double, val x: Double, val y: Double)

/** 최종 궤적: points 의 t 는 영상 기준 절대 시간(초) */
class Trajectory(val impactTime: Double, val points: List<TrackPoint>, val tracked: List<TrackPoint>)

// ---------------------------------------------------------------- 임팩트 감지

class ImpactDetector(ballX: Double, ballY: Double, ballR: Double, private val thr: Double = 18.0) {
    private val cx = ballX.roundToInt()
    private val cy = ballY.roundToInt()
    private val r = max(3, (ballR * 2).roundToInt())
    private var ref: FloatArray? = null
    /** +1: 공이 주변(잔디)보다 밝음, -1: 어두움 */
    var polarity = 1
        private set

    private fun roi(f: GrayFrame): FloatArray {
        val x0 = max(0, cx - r); val x1 = min(f.width, cx + r + 1)
        val y0 = max(0, cy - r); val y1 = min(f.height, cy + r + 1)
        val out = FloatArray(max(0, x1 - x0) * max(0, y1 - y0))
        var i = 0
        for (y in y0 until y1) for (x in x0 until x1) out[i++] = f.at(x, y).toFloat()
        return out
    }

    /** 공 위치가 바뀌면(=임팩트) true */
    fun update(f: GrayFrame): Boolean {
        val cur = roi(f)
        val reference = ref
        if (reference == null || reference.size != cur.size) {
            ref = cur
            if (cur.isNotEmpty()) {
                val center = f.at(cx.coerceIn(0, f.width - 1), cy.coerceIn(0, f.height - 1)).toFloat()
                val sorted = cur.sortedArray()
                polarity = if (center >= sorted[sorted.size / 2]) 1 else -1
            }
            return false
        }
        if (cur.isEmpty()) return false
        var sum = 0.0
        for (i in cur.indices) sum += abs(cur[i] - reference[i])
        val d = sum / cur.size
        if (d < thr * 0.3) { // 조명 변화에 천천히 적응
            for (i in cur.indices) reference[i] = reference[i] * 0.9f + cur[i] * 0.1f
        }
        return d > thr
    }
}

// ---------------------------------------------------------------- 공 추적

class BallTracker(
    private val startX: Double,
    private val startY: Double,
    private val ballR: Double,
    private val w: Int,
    private val h: Int,
    private val polarity: Int = 1,
    private val diffThr: Int = 14,
) {
    companion object {
        const val MAX_GAP_SEC = 0.12
        const val MAX_BUF = 61
        const val MAX_MISS_SEC = 0.2
        const val GIVE_UP_SEC = 0.6   // 임팩트 후 이 시간 안에 공을 못 찾으면 실패
    }

    private val frames = ArrayDeque<GrayFrame>()
    // findBlob 작업용 버퍼(매 프레임 새로 만들지 않고 재사용)
    private var mask = BooleanArray(0)
    private var db = IntArray(0)
    private var visited = BooleanArray(0)
    private var stack = IntArray(0)
    val pts = ArrayList<TrackPoint>()
    private var miss = 0
    private var firstTime = Double.NaN
    var done = false
        private set

    // 최근 4개 검출점의 선형 회귀 기울기 = 순간 속도(px/s)
    private fun velocity(): DoubleArray {
        val n = min(4, pts.size)
        if (n < 2) return doubleArrayOf(0.0, 0.0)
        val q = pts.subList(pts.size - n, pts.size)
        val tLast = q.last().t
        if (tLast - q.first().t < 1e-6) return doubleArrayOf(0.0, 0.0)
        val tt = DoubleArray(n) { q[it].t - tLast }
        return doubleArrayOf(
            linearSlope(tt, DoubleArray(n) { q[it].x }),
            linearSlope(tt, DoubleArray(n) { q[it].y }),
        )
    }

    private fun speed(): Double { val v = velocity(); return hypot(v[0], v[1]) }

    private fun predict(t: Double): DoubleArray {
        val last = pts.last(); val v = velocity(); val dt = t - last.t
        return doubleArrayOf(last.x + v[0] * dt, last.y + v[1] * dt)
    }

    private fun frameDt(): Double {
        if (frames.size < 2) return 1.0 / 30
        return max(frames.last().timeSec - frames[frames.size - 2].timeSec, 1e-4)
    }

    private fun maxGap(): Int {
        if (frames.size < 2) return 1
        return max(1, min((MAX_BUF - 1) / 2, (MAX_GAP_SEC / frameDt()).roundToInt()))
    }

    /** 공이 프레임 간에 자기 크기만큼은 움직이도록 차분 간격(프레임 수)을 조절 */
    private fun gap(): Int {
        if (pts.size < 2 || frames.size < 2) return 1
        val v = speed() * frameDt()
        val need = 2 * ballR + 3
        return min(maxGap(), max(1, ceil(need / max(v, 0.05)).toInt()))
    }

    fun push(f: GrayFrame) {
        if (done) return
        if (firstTime.isNaN()) firstTime = f.timeSec
        frames.addLast(f)
        while (frames.size > MAX_BUF) frames.removeFirst()
        if (pts.isEmpty() && f.timeSec - firstTime > GIVE_UP_SEC) { done = true; return }

        val g = gap()
        if (frames.size < 2 * g + 1) return
        val a = frames[frames.size - 1 - 2 * g]
        val b = frames[frames.size - 1 - g]
        val c = frames.last()
        val tb = b.timeSec
        if (pts.isNotEmpty() && tb <= pts.last().t) return

        var bx0: Double; var bx1: Double; var by0: Double; var by1: Double
        var pred: DoubleArray? = null
        var rad = 0.0
        if (pts.size < 2) { // 티 위쪽(또는 첫 검출점 위쪽) 넓은 영역 탐색
            val ax = if (pts.isEmpty()) startX else pts.last().x
            val ay = if (pts.isEmpty()) startY else pts.last().y
            bx0 = ax - w * 0.3; bx1 = ax + w * 0.3; by0 = ay - h * 0.6; by1 = ay
            if (pts.isNotEmpty() && tb - pts.last().t > 0.1) pts.clear()
        } else {
            val last = pts.last()
            val full = predict(tb)
            val p = doubleArrayOf(last.x + 0.75 * (full[0] - last.x), last.y + 0.75 * (full[1] - last.y))
            val step = hypot(full[0] - last.x, full[1] - last.y)
            val base = max(8.0, step * 0.45 + 6)
            rad = min(base + miss, 2 * base)
            pred = p
            bx0 = p[0] - rad; bx1 = p[0] + rad; by0 = p[1] - rad; by1 = p[1] + rad
        }
        val x0 = max(0, bx0.toInt()); val x1 = min(w, (bx1 + 1).toInt())
        val y0 = max(0, by0.toInt()); val y1 = min(h, (by1 + 1).toInt())
        if (x1 <= x0 || y1 <= y0) { onMiss(); return }

        val best = findBlob(a, b, c, x0, x1, y0, y1, pred, rad)
        if (best == null) onMiss() else { pts.add(TrackPoint(tb, best[0], best[1])); miss = 0 }
    }

    /** 3프레임 차분 마스크에서 연결 성분을 찾고 공 후보 중 가장 가까운 것을 반환 */
    private fun findBlob(
        a: GrayFrame, b: GrayFrame, c: GrayFrame,
        x0: Int, x1: Int, y0: Int, y1: Int, pred: DoubleArray?, rad: Double,
    ): DoubleArray? {
        val ww = x1 - x0; val hh = y1 - y0
        val size = ww * hh
        if (mask.size < size) { mask = BooleanArray(size); db = IntArray(size); visited = BooleanArray(size); stack = IntArray(size) }
        val mask = mask; val db = db; val visited = visited; val stack = stack
        java.util.Arrays.fill(visited, 0, size, false)
        for (y in y0 until y1) {
            val row = (y - y0) * ww
            for (x in x0 until x1) {
                val vb = b.at(x, y)
                val d1 = vb - a.at(x, y)
                val d2 = c.at(x, y) - vb
                db[row + x - x0] = d1
                mask[row + x - x0] = abs(d1) > diffThr && abs(d2) > diffThr
            }
        }
        val maxArea = max(30.0, (ballR * 4) * (ballR * 4))
        var best: DoubleArray? = null
        var bestD = Double.MAX_VALUE
        val refX = pred?.get(0) ?: (if (pts.isEmpty()) startX else pts.last().x)
        val refY = pred?.get(1) ?: (if (pts.isEmpty()) startY else pts.last().y)
        for (i0 in 0 until size) {
            if (!mask[i0] || visited[i0]) continue
            var sp = 0; stack[sp++] = i0; visited[i0] = true
            var area = 0; var sx = 0.0; var sy = 0.0; var sd = 0.0
            while (sp > 0) {
                val i = stack[--sp]
                val px = i % ww; val py = i / ww
                area++; sx += px; sy += py; sd += db[i]
                for (dy in -1..1) for (dx in -1..1) {
                    if (dx == 0 && dy == 0) continue
                    val nx = px + dx; val ny = py + dy
                    if (nx < 0 || ny < 0 || nx >= ww || ny >= hh) continue
                    val j = ny * ww + nx
                    if (mask[j] && !visited[j]) { visited[j] = true; stack[sp++] = j }
                }
            }
            if (area > maxArea) continue
            val cxp = sx / area + x0; val cyp = sy / area + y0
            if (pts.size < 3) {
                if (cyp > startY - ballR * 1.5) continue          // 초기 검출은 티보다 위
                if ((sd / area) * polarity <= 0) continue          // 클럽/몸: 밝기 방향이 반대
            }
            val d = hypot(cxp - refX, cyp - refY)
            if (pred != null && d > rad) continue
            if (d < bestD) { bestD = d; best = doubleArrayOf(cxp, cyp) }
        }
        return best
    }

    private fun onMiss() {
        if (pts.isEmpty()) return
        miss++
        if (frames.last().timeSec - pts.last().t > MAX_MISS_SEC + MAX_GAP_SEC) done = true
    }
}

// ---------------------------------------------------------------- 궤적 피팅

object TrajectoryFitter {

    /**
     * 공기저항을 포함한 원근 투영 탄도 모델
     *   u(t) = (A0 + A1·φ + A2·t²) / (1 + a·φ),  v(t) = (B0 + B1·φ + B2·t²) / (1 + a·φ)
     *   φ(t) = (1 - e^{-kt}) / k    (k 는 격자 탐색, 나머지는 선형 최소제곱)
     * 추적이 짧게 끊겨도 나머지 비행을 예측해서 그린다.
     */
    fun fit(
        startX: Double, startY: Double, raw: List<TrackPoint>, w: Int, h: Int,
        landingFraction: Double = 0.3, tMax: Double = 8.0, dt: Double = 1.0 / 120,
    ): Trajectory? {
        if (raw.size < 3) return null
        val pts = removeOutliers(raw)
        if (pts.size < 3) return null
        // 임팩트 시각 추정: 처음 3개 점의 평균 속도로 티까지 거꾸로 외삽.
        // 초반 두 점이 노이즈로 겹쳐도 발산하지 않도록 0.25초 이내로 제한
        val k3 = min(2, pts.size - 1)
        val step = hypot(pts[k3].x - pts[0].x, pts[k3].y - pts[0].y) / max(1e-6, pts[k3].t - pts[0].t)
        val back = hypot(pts[0].x - startX, pts[0].y - startY) / max(step, 1e-6)
        val t0 = pts[0].t - back.coerceIn(0.0, 0.25)
        val n = pts.size + 1
        val t = DoubleArray(n); val u = DoubleArray(n); val v = DoubleArray(n); val wt = DoubleArray(n) { 1.0 }
        t[0] = 0.0; u[0] = startX; v[0] = startY; wt[0] = 4.0
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

        val out = ArrayList<TrackPoint>()
        var apexV = Double.NaN; var apexSeen = false
        var tt = 0.0
        while (tt <= tMax) {
            val m = fn(tt)
            if (m[2] < 0.05) break
            val x = m[0] / m[2]; val y = m[1] / m[2]
            if (apexV.isNaN() || y < apexV) apexV = y
            else if (y > apexV + 1) apexSeen = true
            if (apexSeen && y >= apexV + landingFraction * (startY - apexV)) { out.add(TrackPoint(t0 + tt, x, y)); break }
            if (x < -w || x > 2.0 * w || y < -h || y > 2.0 * h) break
            out.add(TrackPoint(t0 + tt, x, y))
            tt += dt
        }
        return Trajectory(t0, out, pts)
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

// ---------------------------------------------------------------- 수치 계산 도우미

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
    for (j in 0 until m) scale[j] = if (scale[j] > 0) Math.sqrt(scale[j]) else 1.0
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
