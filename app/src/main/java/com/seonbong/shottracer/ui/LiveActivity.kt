package com.seonbong.shottracer.ui

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.hardware.camera2.CameraCharacteristics
import android.os.Bundle
import android.util.Range
import android.view.Gravity
import android.view.MotionEvent
import android.view.WindowManager
import android.widget.Button
import android.widget.FrameLayout
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.UseCaseGroup
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.seonbong.shottracer.engine.GrayFrame
import com.seonbong.shottracer.engine.ShotSession
import java.util.concurrent.Executors

/**
 * 준실시간 모드: 미리보기 화면에서 공을 탭 → 스윙 → 공이 날아가는 동안 점으로 추적 →
 * 추적이 끝나면 예측 궤적(빨간 선)을 미리보기 위에 그린다. 폰은 반드시 고정.
 */
class LiveActivity : ComponentActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var overlay: TraceOverlayView
    private val analysisExecutor = Executors.newSingleThreadExecutor()

    // 분석 스레드에서만 접근
    private var session: ShotSession? = null
    @Volatile private var pendingTap: Pair<Float, Float>? = null
    @Volatile private var lastFrameSize = Pair(0, 0)
    private var fpsInfo = ""

    private val permission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
        if (ok) startCamera() else { Toast.makeText(this, "카메라 권한이 필요합니다", Toast.LENGTH_LONG).show(); finish() }
    }

    @SuppressLint("ClickableViewAccessibility")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        val root = FrameLayout(this)
        previewView = PreviewView(this).apply { scaleType = PreviewView.ScaleType.FIT_CENTER }
        overlay = TraceOverlayView(this).apply { message = "폰을 고정하고, 화면에서 공을 탭하세요" }
        val reset = Button(this).apply {
            text = "다시 설정"
            setOnClickListener {
                analysisExecutor.execute { session = null }
                // Button.apply 안에서는 'overlay'가 View 자체 속성으로 해석되므로 명시
                val ov = this@LiveActivity.overlay
                ov.ball = null; ov.tracked = emptyList(); ov.curve = emptyList()
                ov.message = "공을 탭하세요"; ov.invalidate()
            }
        }
        root.addView(previewView, FrameLayout.LayoutParams(-1, -1))
        root.addView(overlay, FrameLayout.LayoutParams(-1, -1))
        root.addView(reset, FrameLayout.LayoutParams(-2, -2, Gravity.BOTTOM or Gravity.END).apply { setMargins(0, 0, 32, 48) })
        setContentView(root)

        overlay.setOnTouchListener { _, e ->
            if (e.action == MotionEvent.ACTION_UP) pendingTap = Pair(e.x, e.y)
            true
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) startCamera()
        else permission.launch(Manifest.permission.CAMERA)
    }

    @androidx.annotation.OptIn(ExperimentalCamera2Interop::class)
    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val selector = CameraSelector.DEFAULT_BACK_CAMERA
            val ratio = ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY).build()

            // 지원되는 가장 높은 고정 프레임레이트 선택 (보통 30 또는 60)
            val fps = try {
                val info = provider.availableCameraInfos.firstOrNull { it.lensFacing == CameraSelector.LENS_FACING_BACK }
                val ranges = info?.let {
                    Camera2CameraInfo.from(it).getCameraCharacteristic(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES)
                }
                ranges?.maxWithOrNull(compareBy<Range<Int>>({ it.upper }, { it.lower }))
            } catch (e: Exception) { null }
            fpsInfo = fps?.let { "${it.upper}fps" } ?: ""

            val previewBuilder = Preview.Builder().setResolutionSelector(ratio)
            if (fps != null) previewBuilder.setTargetFrameRate(fps)
            val preview = previewBuilder.build().also { it.setSurfaceProvider(previewView.surfaceProvider) }

            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(ratio)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
            analysis.setAnalyzer(analysisExecutor) { image ->
                try { onFrame(FrameConvert.fromImageProxy(image)) } finally { image.close() }
            }
            // 미리보기와 분석이 "같은 화면 영역"을 보도록 ViewPort로 묶는다.
            // (기기가 둘에 서로 다른 비율을 골라도 분석 프레임은 cropRect로 미리보기와 맞춰짐)
            previewView.post {
                if (isDestroyed || isFinishing) return@post   // 카메라 준비 전에 화면을 닫은 경우
                val group = UseCaseGroup.Builder().addUseCase(preview).addUseCase(analysis)
                previewView.viewPort?.let { group.setViewPort(it) }
                provider.unbindAll()
                provider.bindToLifecycle(this, selector, group.build())
            }
        }, ContextCompat.getMainExecutor(this))
    }

    /** 분석 스레드 */
    private fun onFrame(f: GrayFrame) {
        if (lastFrameSize != Pair(f.width, f.height)) {
            lastFrameSize = Pair(f.width, f.height)
            runOnUiThread { overlay.frameWidth = f.width; overlay.frameHeight = f.height; overlay.invalidate() }
        }
        pendingTap?.let { (vx, vy) ->
            pendingTap = null
            val p = overlay.toFrame(vx, vy) ?: return@let
            val s = ShotSession(f, p.first, p.second)   // 탭 위치를 실제 공 중심으로 보정
            session = s
            runOnUiThread {
                overlay.ball = Pair(s.ball.x, s.ball.y); overlay.ballRadius = s.ballR
                overlay.tracked = emptyList(); overlay.curve = emptyList()
                overlay.message = "준비 완료 $fpsInfo — 스윙하세요"; overlay.invalidate()
            }
        }
        val s = session ?: return
        val changed = s.feed(f)
        when (s.state) {
            ShotSession.State.TRACKING -> {
                val pts = s.trackedPoints
                runOnUiThread { overlay.tracked = pts; overlay.message = "추적 중… (${pts.size})"; overlay.invalidate() }
            }
            ShotSession.State.FINISHED -> if (changed) {
                val tr = s.trajectory!!
                session = null
                runOnUiThread { animateCurve(tr.points) }
            }
            ShotSession.State.FAILED -> if (changed) {
                session = null
                runOnUiThread {
                    overlay.message = "공을 찾지 못했어요. 공을 다시 탭하세요"
                    overlay.ball = null; overlay.tracked = emptyList(); overlay.invalidate()
                }
            }
            else -> {}
        }
    }

    /** 궤적을 1.5초 동안 그려 나가는 애니메이션 */
    private fun animateCurve(points: List<com.seonbong.shottracer.engine.TrackPoint>) {
        if (points.isEmpty()) return
        overlay.curve = points
        overlay.tracked = emptyList()
        overlay.message = "다음 샷: 공을 탭하세요"
        val t0 = points.first().t; val t1 = points.last().t
        val start = System.nanoTime()
        val tick = object : Runnable {
            override fun run() {
                val p = ((System.nanoTime() - start) / 1.5e9).coerceAtMost(1.0)
                overlay.revealUntil = t0 + (t1 - t0) * p
                overlay.invalidate()
                if (p < 1.0) overlay.postOnAnimation(this) else overlay.revealUntil = Double.MAX_VALUE
            }
        }
        overlay.postOnAnimation(tick)
    }

    override fun onDestroy() {
        super.onDestroy()
        analysisExecutor.shutdown()
    }
}
