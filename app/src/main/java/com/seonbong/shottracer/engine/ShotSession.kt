package com.seonbong.shottracer.engine

/**
 * 한 번의 샷을 처리하는 상태 머신. 실시간 모드와 갤러리 모드가 같이 쓴다.
 *   ARMED(임팩트 대기) → TRACKING(공 포착·추적) → FINISHED(궤적 완성) / FAILED
 * 공이 날아가는 걸 못 찾았는데 공이 아직 티에 있으면(오감지) 다시 ARMED 로 돌아간다.
 *
 * @param first 사용자가 공을 탭한 프레임
 */
class ShotSession(first: GrayFrame, tapX: Double, tapY: Double) {
    enum class State { ARMED, TRACKING, FINISHED, FAILED }

    companion object { const val GIVE_UP_SEC = 0.6; const val HISTORY = 64 }

    /** 탭 위치를 실제 공 중심으로 보정한 결과 */
    val ball: BallInfo = refineBall(first, tapX, tapY)
    val ballR: Double get() = ball.r

    var state = State.ARMED
        private set
    var trajectory: Trajectory? = null
        private set
    /** 오감지 후 다시 대기한 횟수(디버그용) */
    var rearms = 0
        private set

    private val w = first.width
    private val h = first.height
    private var detector = ImpactDetector(ball)
    private var tracker: BallTracker? = null
    private var impactTime = Double.NaN
    private val history = ArrayDeque<GrayFrame>()

    val trackedPoints: List<TrackPoint> get() = tracker?.pts?.toList() ?: emptyList()

    /** 프레임을 시간 순서대로 넣는다. 상태가 바뀌면 true */
    fun feed(f: GrayFrame): Boolean {
        val before = state
        history.addLast(f)
        while (history.size > HISTORY) history.removeFirst()
        when (state) {
            State.ARMED -> if (detector.update(f)) {
                impactTime = detector.impactTime
                val t = BallTracker(ball.x, ball.y, ball.r, w, h, ball.polarity)
                // 공이 사라지기 직전 프레임부터 다시 넣어 준다
                val dt = if (history.size > 1) (f.timeSec - history.first().timeSec) / (history.size - 1) else 1.0 / 30
                for (hf in history) if (hf.timeSec >= impactTime - 1.5 * dt) t.push(hf)
                tracker = t
                state = State.TRACKING
            }
            State.TRACKING -> {
                val t = tracker!!
                t.push(f)
                if (!t.acquired && (t.done || f.timeSec - impactTime > GIVE_UP_SEC)) {
                    if (detector.ballPresent(f)) {   // 오감지: 공은 아직 티 위에 있다
                        detector = ImpactDetector(ball)
                        detector.update(f)
                        tracker = null
                        rearms++
                        state = State.ARMED
                    } else {
                        state = State.FAILED
                    }
                } else if (t.done) {
                    finish()
                }
            }
            else -> {}
        }
        return state != before
    }

    /** 영상이 끝났거나 추적이 끝났을 때 지금까지의 추적으로 궤적을 만든다 */
    fun finish(@Suppress("UNUSED_PARAMETER") width: Int = w, @Suppress("UNUSED_PARAMETER") height: Int = h) {
        if (state != State.TRACKING) return
        val pts = tracker?.pts ?: emptyList()
        trajectory = if (pts.size >= 3) TrajectoryFitter.fit(ball.x, ball.y, pts, w, h) else null
        state = if (trajectory != null) State.FINISHED else State.FAILED
    }
}
