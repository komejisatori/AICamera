package com.ufo.aicamera;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;

import java.util.ArrayList;

public class Bbox {

    private final Paint painter = new Paint();
    private ArrayList<Float> trackingRects = new ArrayList<Float>();


    public Bbox(final Context context){
        painter.setColor(Color.GREEN);
        painter.setStyle(Paint.Style.STROKE);
        painter.setStrokeWidth(10.0f);
        painter.setStrokeCap(Paint.Cap.ROUND);
        painter.setStrokeJoin(Paint.Join.ROUND);
        painter.setStrokeMiter(100);

    }

    // TODO: a method for adding tracking results to tracker as in the tflite "processResult" method

    public synchronized void draw(final Canvas canvas ){//, ArrayList<float> prob, ArrayList<float> index){

        if (trackingRects.isEmpty()) {
            return;
        }

        /*
        for (int i =0; i <= trackingRects.size(); i+=4){
            final RectF bboxPos = new RectF(trackingRects.get(i), trackingRects.get(i+1), trackingRects.get(i+2), trackingRects.get(i + 3));
            canvas.drawRect(bboxPos, painter);
        }
        */
        final RectF bboxPos = new RectF(0, 0, 100, 100);
        canvas.drawRect(bboxPos, painter);

    }

    public void trackResults(final ArrayList<Float> results){
        trackingRects = results;
    }
}
