package com.seonbong.shottracer.ui

import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.graphics.RectF
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Bundle
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.widget.Button
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.SeekBar
import android.widget.TextView
import android.widget.VideoView
import androidx.activity.ComponentActivity
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import com.seonbong.shottracer.engine.ShotSession
import com.seonbong.shottracer.engine.TrackPoint
import java.util.concurrent.Executors

/**
 * 갤러리 모드: 저장된 스윙 영상 선택 → 공이 놓여 있는 장면에서 공 탭 → 분석 → 영상 재생 위에 궤적 표시.
 * 슬로모션 영상(120/240fps)일수록 정확하다.
 */
class GalleryActivity : ComponentActivity() {

    private lateinit var image: ImageView
    private lateinit var video: VideoView
    private lateinit var overlay: TraceOverlayView
    private lateinit var seek: SeekBar
    private lateinit var analyzeBtn: Button
    private lateinit var replayBtn: Button
    private lateinit var info: TextView
    private val worker = Executors.newSingleThreadExecutor()

    private var uri: Uri? = null
    private var frameCount = 0
    private var durationSec = 0.0
    /** 영상 시간 → 실제 시간 배율 (슬로모션 파일이 30fps로 재생되도록 저장된 경우 보정) */
    private var timeScale = 1.0
    @Volatile private var startIndex = 0
    @Volatile private var cancelled = false
    /** 프레임별 실제 표시 시각(초). 휴대폰 영상은 가변 프레임레이트(VFR)인 경우가 많아 인덱스로 계산하지 않는다 */
    private var frameTimes = DoubleArray(0)
    private var ball: Pair<Double, Double>? = null
    /** 화면에 보이는 프레임(공 탭 위치 보정용) */
    @Volatile private var shownGray: com.seonbong.shottracer.engine.GrayFrame? = null
    private var curve: List<TrackPoint> = emptyList()

    private val picker = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { u ->
        if (u == null) finish() else load(u)
    }

    @SuppressLint("ClickableViewAccessibility")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val stage = FrameLayout(this)
        image = ImageView(this).apply { scaleType = ImageView.ScaleType.FIT_CENTER }
        video = VideoView(this).apply { visibility = View.GONE }
        overlay = TraceOverlayView(this)
        stage.addView(image, FrameLayout.LayoutParams(-1, -1))
        stage.addView(video, FrameLayout.LayoutParams(-2, -2, Gravity.CENTER))
        stage.addView(overlay, FrameLayout.LayoutParams(-1, -1))

        info = TextView(this).apply { setPadding(24, 12, 24, 0) }
        seek = SeekBar(this)
        analyzeBtn = Button(this).apply { text = "궤적 분석"; isEnabled = false; setOnClickListener { analyze() } }
        replayBtn = Button(this).apply { text = "다시 재생"; isEnabled = false; setOnClickListener { play() } }
        val other = Button(this).apply { text = "다른 영상"; setOnClickListener { pick() } }
        val buttons = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER
            addView(analyzeBtn); addView(replayBtn); addView(other)
        }
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(stage, LinearLayout.LayoutParams(-1, 0, 1f))
            addView(info); addView(seek); addView(buttons)
        }
        setContentView(root)

        overlay.setOnTouchListener { _, e ->
            if (e.action == MotionEvent.ACTION_UP && image.visibility == View.VISIBLE) {
                overlay.toFrame(e.x, e.y)?.let { p ->
                    ball = p
                    // 탭 위치를 실제 공 중심으로 보정해서 보여 준다
                    val info = shownGray?.let { com.seonbong.shottracer.engine.refineBall(it, p.first, p.second) }
                    overlay.ball = if (info != null) Pair(info.x, info.y) else p
                    overlay.ballRadius = info?.r ?: maxOf(2.0, overlay.frameWidth * 0.01)
                    overlay.message = "공 위치 설정됨 → [궤적 분석]"
                    analyzeBtn.isEnabled = true
                    overlay.invalidate()
                }
            }
            true
        }
        seek.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar, p: Int, fromUser: Boolean) { if (fromUser) showFrame(p) }
            override fun onStartTrackingTouch(sb: SeekBar) {}
            override fun onStopTrackingTouch(sb: SeekBar) {}
        })
        video.addOnLayoutChangeListener { v, l, t, r, b, _, _, _, _ ->
            overlay.contentRect = RectF(l.toFloat(), t.toFloat(), r.toFloat(), b.toFloat())
            overlay.invalidate()
        }
        pick()
    }

    private fun pick() = picker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.VideoOnly))

    private fun retriever(): MediaMetadataRetriever = MediaMetadataRetriever().apply { setDataSource(this@GalleryActivity, uri!!) }

    private fun load(u: Uri) {
        uri = u
        ball = null; curve = emptyList()
        video.stopPlayback(); video.visibility = View.GONE; image.visibility = View.VISIBLE
        overlay.contentRect = null; overlay.ball = null; overlay.curve = emptyList(); overlay.tracked = emptyList()
        analyzeBtn.isEnabled = false; replayBtn.isEnabled = false
        worker.execute {
            frameTimes = readFrameTimes()
            val r = retriever()
            try {
                // 프레임 수 메타데이터가 없는 파일도 있어서 실제 샘플 수를 우선 사용
                frameCount = if (frameTimes.isNotEmpty()) frameTimes.size
                    else r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)?.toIntOrNull() ?: 0
                durationSec = (r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toDoubleOrNull() ?: 0.0) / 1000
                if (frameCount == 0) {
                    runOnUiThread { info.text = "이 영상은 프레임 정보를 읽을 수 없어요. 다른 영상을 골라 주세요" }
                    return@execute
                }
                val fileFps = if (durationSec > 0) frameCount / durationSec else 30.0
                val captureFps = r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_CAPTURE_FRAMERATE)?.toDoubleOrNull() ?: 0.0
                timeScale = if (captureFps > fileFps * 1.5) fileFps / captureFps else 1.0
                val realFps = if (timeScale < 1.0) captureFps else fileFps
                runOnUiThread {
                    seek.max = maxOf(0, frameCount - 1)
                    info.text = "프레임 ${frameCount}개 · 약 ${"%.0f".format(realFps)}fps · 아래 막대로 공이 놓여 있는 장면을 고르고 공을 탭하세요"
                }
            } finally { r.release() }
            showFrame(0)
        }
    }

    private fun showFrame(index: Int) {
        startIndex = index
        worker.execute {
            if (startIndex != index) return@execute // 더 최신 요청이 있음
            val r = retriever()
            try {
                val bmp = frameAt(r, index, null) ?: return@execute
                val g = FrameConvert.fromBitmap(bmp, 0.0)
                shownGray = g
                runOnUiThread {
                    image.setImageBitmap(bmp)
                    overlay.frameWidth = g.width; overlay.frameHeight = g.height
                    if (ball == null) overlay.message = "공을 탭하세요"
                    overlay.invalidate()
                }
            } catch (e: Exception) {
                runOnUiThread { info.text = "프레임을 읽을 수 없어요: ${e.message}" }
            } finally { r.release() }
        }
    }

    private fun analyze() {
        val b = ball ?: return
        analyzeBtn.isEnabled = false
        overlay.message = "분석 중…"; overlay.invalidate()
        val from = startIndex
        worker.execute {
            val r = retriever()
            var session: ShotSession? = null
            var w = 0; var h = 0
            try {
                val params = MediaMetadataRetriever.BitmapParams().apply { preferredConfig = Bitmap.Config.RGB_565 }
                var i = from
                val batch = 8
                loop@ while (i < frameCount && !cancelled) {
                    val n = minOf(batch, frameCount - i)
                    val bitmaps: List<Bitmap> = try { r.getFramesAtIndex(i, n, params) }
                        catch (e: Exception) { listOfNotNull(frameAt(r, i, params)) }
                    for ((k, bmp) in bitmaps.withIndex()) {
                        val videoT = timeOf(i + k)
                        val f = FrameConvert.fromBitmap(bmp, videoT * timeScale)
                        bmp.recycle()
                        w = f.width; h = f.height
                        val s = session ?: ShotSession(f, b.first, b.second).also { session = it }
                        s.feed(f)
                        if (s.state == ShotSession.State.FINISHED || s.state == ShotSession.State.FAILED) break@loop
                    }
                    i += maxOf(1, bitmaps.size)
                    val pct = (i - from) * 100 / maxOf(1, frameCount - from)
                    runOnUiThread { overlay.message = "분석 중… $pct%"; overlay.invalidate() }
                }
                session?.finish(w, h)
            } catch (e: Exception) {
                runOnUiThread { info.text = "분석 오류: ${e.message}" }
            } finally { r.release() }

            if (cancelled) return@execute
            val st = session?.state
            val points = session?.trajectory?.points
            runOnUiThread {
                analyzeBtn.isEnabled = true
                if (st == ShotSession.State.FINISHED && points != null) {
                    curve = points
                    replayBtn.isEnabled = true
                    play()
                } else {
                    overlay.message = if (st == ShotSession.State.FAILED) "날아가는 공을 찾지 못했어요"
                        else "임팩트를 찾지 못했어요. 공 위치를 다시 탭해 보세요"
                    overlay.invalidate()
                }
            }
        }
    }

    /** MediaExtractor로 영상 트랙의 실제 샘플 시각을 읽는다(디코딩 없이 빠름). 표시 순서로 정렬 */
    private fun readFrameTimes(): DoubleArray {
        val ex = MediaExtractor()
        return try {
            ex.setDataSource(this, uri!!, null)
            val track = (0 until ex.trackCount).firstOrNull {
                ex.getTrackFormat(it).getString(MediaFormat.KEY_MIME)?.startsWith("video/") == true
            } ?: return DoubleArray(0)
            ex.selectTrack(track)
            val list = ArrayList<Long>()
            while (true) {
                val t = ex.sampleTime
                if (t < 0) break
                list.add(t)
                if (!ex.advance()) break
            }
            list.sort()
            val first = list.firstOrNull() ?: 0L
            DoubleArray(list.size) { (list[it] - first) / 1e6 }
        } catch (e: Exception) {
            DoubleArray(0)
        } finally { ex.release() }
    }

    private fun timeOf(index: Int): Double =
        if (index < frameTimes.size) frameTimes[index] else index * durationSec / maxOf(1, frameCount)

    /** 인덱스로 프레임을 꺼내되, 실패하면 시각 기준으로 가장 가까운 프레임을 꺼낸다 */
    private fun frameAt(r: MediaMetadataRetriever, index: Int, params: MediaMetadataRetriever.BitmapParams?): Bitmap? =
        try {
            if (params != null) r.getFrameAtIndex(index, params) else r.getFrameAtIndex(index)
        } catch (e: Exception) {
            r.getFrameAtTime((timeOf(index) * 1e6).toLong(), MediaMetadataRetriever.OPTION_CLOSEST)
        }

    /** 원본 영상을 재생하면서 재생 위치에 맞춰 궤적을 그려 나간다 */
    private fun play() {
        val u = uri ?: return
        image.visibility = View.GONE
        video.visibility = View.VISIBLE
        overlay.ball = null
        overlay.curve = curve
        overlay.message = ""
        overlay.revealUntil = Double.NEGATIVE_INFINITY
        video.setVideoURI(u)
        video.setOnCompletionListener { overlay.revealUntil = Double.MAX_VALUE; overlay.invalidate() }
        video.setOnPreparedListener { video.start(); overlay.postOnAnimation(tick) }
    }

    private val tick = object : Runnable {
        override fun run() {
            if (video.visibility != View.VISIBLE) return
            if (video.isPlaying) {
                overlay.revealUntil = video.currentPosition / 1000.0 * timeScale
                overlay.invalidate()
                overlay.postOnAnimation(this)
            } else if (overlay.revealUntil != Double.MAX_VALUE) {
                overlay.postOnAnimation(this)
            }
        }
    }

    override fun onDestroy() {
        cancelled = true   // 분석 루프 즉시 중단
        super.onDestroy()
        worker.shutdown()
    }
}
