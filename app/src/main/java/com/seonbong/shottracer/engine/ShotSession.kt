package com.seonbong.shottracer.engine

import kotlin.math.max

/**
 * 한 번의 샷을 처리하는 상태 머신. 실시간 모드와 갤러리 모드가 같이 쓴다.
 *   ARMED(임팩트 대기) → TRACKING(공 추적) → FINISHED(궤적 완성) / FAILED(공을 못 찾음)
 */
class ShotSession(
    private val ballX: Double,
    private val ballY: Double,
    frameWidth: Int,
) {
    enum class State { ARMED, TRACKING, FINISHED, FAILED }

    /** 공 반지름 추정값(분석 프레임 기준 px) */
    val ballR = max(2.0, frameWidth * 0.01)

    var state = State.ARMED
        private set
    var trajectory: Trajectory? = null
        private set

    private val detector = ImpactDetector(ballX, ballY, ballR)
    private var tracker: BallTracker? = null
    private var prev: GrayFrame? = null

    val trackedPoints: List<TrackPoint> get() = tracker?.pts?.toList() ?: emptyList()

    /** 프레임을 순서대로 넣는다. 상태가 바뀌면 true */
    fun feed(f: GrayFrame): Boolean {
        val before = state
        when (state) {
            State.ARMED -> {
                if (detector.update(f)) {
                    val t = BallTracker(ballX, ballY, ballR, f.width, f.height, detector.polarity)
                    prev?.let { t.push(it) }
                    t.push(f)
                    tracker = t
                    state = State.TRACKING
                }
                prev = f
            }
            State.TRACKING -> {
                val t = tracker!!
                t.push(f)
                if (t.done) finish(f.width, f.height)
            }
            else -> {}
        }
        return state != before
    }

    /** 영상이 끝났거나 사용자가 멈췄을 때 지금까지의 추적으로 궤적을 만든다 */
    fun finish(w: Int, h: Int) {
        if (state != State.TRACKING) return
        val pts = tracker?.pts ?: emptyList()
        trajectory = if (pts.size >= 3) TrajectoryFitter.fit(ballX, ballY, pts, w, h) else null
        state = if (trajectory != null) State.FINISHED else State.FAILED
    }
}
