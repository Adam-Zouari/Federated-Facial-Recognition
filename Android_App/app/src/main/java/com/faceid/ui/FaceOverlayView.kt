package com.faceid.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View

class FaceOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val paint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 8f
    }

    private val textPaint = Paint().apply {
        color = Color.GREEN
        textSize = 48f
        textAlign = Paint.Align.CENTER
    }

    private var faceRect: RectF? = null
    private var showNoFaceMessage = false

    fun setFaceRect(rect: RectF?) {
        faceRect = rect
        showNoFaceMessage = rect == null
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        faceRect?.let { rect ->
            // Draw green rectangle around face
            canvas.drawRect(rect, paint)
            
            // Draw "Face Detected" text
            canvas.drawText(
                "Face Detected ✓",
                width / 2f,
                rect.top - 20f,
                textPaint
            )
        }

        if (showNoFaceMessage && faceRect == null) {
            // Draw "No Face" message
            textPaint.color = Color.RED
            canvas.drawText(
                "No Face Detected",
                width / 2f,
                height / 2f,
                textPaint
            )
            textPaint.color = Color.GREEN
        }
    }
}
