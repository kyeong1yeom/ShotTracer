package com.seonbong.shottracer

import android.content.Intent
import android.os.Bundle
import android.view.Gravity
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.ComponentActivity
import com.seonbong.shottracer.ui.GalleryActivity
import com.seonbong.shottracer.ui.LiveActivity

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val title = TextView(this).apply {
            text = "샷 트레이서"; textSize = 32f; gravity = Gravity.CENTER
        }
        val tip = TextView(this).apply {
            text = "폰을 삼각대에 고정하고, 골퍼 뒤에서 목표 방향으로 찍어 주세요.\n슬로모션(120/240fps) 영상이 가장 정확합니다."
            gravity = Gravity.CENTER; setPadding(48, 24, 48, 48)
        }
        val live = Button(this).apply {
            text = "실시간 촬영 모드"
            setOnClickListener { startActivity(Intent(this@MainActivity, LiveActivity::class.java)) }
        }
        val gallery = Button(this).apply {
            text = "저장된 영상 분석"
            setOnClickListener { startActivity(Intent(this@MainActivity, GalleryActivity::class.java)) }
        }
        setContentView(LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL; gravity = Gravity.CENTER; setPadding(64, 64, 64, 64)
            addView(title); addView(tip); addView(live); addView(gallery)
        })
    }
}
