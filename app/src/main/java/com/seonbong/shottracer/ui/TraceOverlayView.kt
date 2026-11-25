package com.seonbong.shottracer.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.view.View
import com.seonbong.shottracer.engine.TrackPoint
import kotlin.math.min

/**
 * 카메라 미리보기/영상 위에 궤적을 그리는 투명 뷰.
 * 엔진 좌표(분석 프레임 픽셀) → 화면 좌표 변환은 contentRect(영상이 실제로 보이는 영역) 기준.
 */
class TraceOverlayView(context: Context) : View(context) {

    var frameWidth = 1
    var frameHeight = 1
    /** 영상이 그려지는 영역. null 이면 뷰 안에 FIT_CENTER 로 맞춘 영역을 사용 */
    var contentRect: RectF? = null

    var ball: Pair<Double, Double>? = null
    var ballRadius = 0.0
    var tracked: List<TrackPoint> = emptyList()
    var curve: List<TrackPoint> = emptyList()
    /** curve 중 이 시간(엔진 기준)까지만 그림. Double.MAX_VALUE 면 전체 */
    var revealUntil = Double.MAX_VALUE
    var message: String = ""

    private val curvePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.rgb(255, 59, 48); style = Paint.Style.STROKE
        strokeWidth = 10f; strokeCap = Paint.Cap.ROUND; strokeJoin = Paint.Join.ROUND
    }
    private val glowPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(90, 255, 200, 0); style = Paint.Style.STROKE
        strokeWidth = 22f; strokeCap = Paint.Cap.ROUND; strokeJoin = Paint.Join.ROUND
    }
    private val dotPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.rgb(0, 200, 255) }
    private val ballPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.YELLOW; style = Paint.Style.STROKE; strokeWidth = 4f
    }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE; textSize = 44f; setShadowLayer(6f, 0f, 0f, Color.BLACK)
    }

    fun rect(): RectF {
        contentRect?.let { return it }
        val s = min(width.toFloat() / frameWidth, height.toFloat() / frameHeight)
        val w = frameWidth * s; val h = frameHeight * s
        return RectF((width - w) / 2, (height - h) / 2, (width + w) / 2, (height + h) / 2)
    }

    private fun sx(r: RectF, x: Double) = (r.left + x / frameWidth * r.width()).toFloat()
    private fun sy(r: RectF, y: Double) = (r.top + y / frameHeight * r.height()).toFloat()

    /** 화면 터치 좌표 → 프레임 좌표 (영상 밖이면 null) */
    fun toFrame(vx: Float, vy: Float): Pair<Double, Double>? {
        val r = rect()
        if (!r.contains(vx, vy)) return null
        return Pair(((vx - r.left) / r.width() * frameWidth).toDouble(), ((vy - r.top) / r.height() * frameHeight).toDouble())
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val r = rect()
        ball?.let { (bx, by) ->
            val rr = (ballRadius / frameWidth * r.width()).toFloat() * 2.5f + 12f
            canvas.drawCircle(sx(r, bx), sy(r, by), rr, ballPaint)
        }
        val shown = curve.filter { it.t <= revealUntil }
        if (shown.size >= 2) {
            val path = Path()
            path.moveTo(sx(r, shown[0].x), sy(r, shown[0].y))
            for (p in shown.drop(1)) path.lineTo(sx(r, p.x), sy(r, p.y))
            canvas.drawPath(path, glowPaint)
            canvas.drawPath(path, curvePaint)
        } else {
            for (p in tracked) canvas.drawCircle(sx(r, p.x), sy(r, p.y), 6f, dotPaint)
        }
        if (message.isNotEmpty()) canvas.drawText(message, r.left + 32f, r.top + 80f, textPaint)
    }
}
