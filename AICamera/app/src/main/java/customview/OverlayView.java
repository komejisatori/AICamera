package customview;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;


import com.ufo.aicamera.Recognition;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import static java.lang.Math.max;

/** A simple View providing a render callback to other classes. */
public class OverlayView extends View {

    private final Paint painter = new Paint();
    private final Paint writer = new Paint();
    // private ArrayList<Float> trackingRects = new ArrayList<Float>();
    private ArrayList<Recognition> trackingResults = new ArrayList<Recognition>();
    //private final Float offset_h = 16.0;
    //private final Float offset_w = 96.0;

    public OverlayView(final Context context, final AttributeSet attrs) {

        super(context, attrs);

        painter.setColor(Color.GREEN);
        painter.setStyle(Paint.Style.STROKE);
        painter.setStrokeWidth(10.0f);
        painter.setStrokeCap(Paint.Cap.ROUND);
        painter.setStrokeJoin(Paint.Join.ROUND);
        painter.setStrokeMiter(100);

        writer.setColor(Color.BLUE);
        writer.setTextSize(40f);
    }

    public void setRects(ArrayList<Recognition> rects){

        for (Recognition p: rects){
            trackingResults.add(p);
        }
        // trackingRects = rects;
        // invalidate(0, 0, this.getWidth(), this.getHeight());
    }

    @Override
    protected void onDraw(final Canvas canvas) {
        super.onDraw(canvas);


        if (trackingResults.isEmpty()) {
            return;
        }

        for (Recognition reco: trackingResults){
            RectF bboxPos = reco.bbox;
            final RectF bbox = new RectF( (480 - (bboxPos.right + 16)) * 2f,
                    (bboxPos.top + 96)* 2f,
                    (480 - (bboxPos.left + 16))*2f,
                    (bboxPos.bottom + 96)* 2f);
            canvas.drawRect(bbox, painter);
            canvas.drawText( String.format("%s : %f", reco.cls, reco.prob), bbox.left, bbox.top,
                    writer);
        }

        /*
        for (int i =0; i < trackingRects.size(); i+=4){
            final RectF bboxPos = new RectF((480- (trackingRects.get(i) + 16)) * 2f, (trackingRects.get(i+1) + 96)*2f, (480 - (trackingRects.get(i+2)+16))*2f, (trackingRects.get(i + 3)+96)*2f);
            //final RectF bboxPos_ = new RectF(0,0,500,500);
            canvas.drawRect(bboxPos, painter);

            //canvas.drawRect(bboxPos_, painter);
        }
        */
        trackingResults.clear();
        /*

        final RectF bboxPos = new RectF(0, 0, 100, 500);
        canvas.drawRect(bboxPos, painter);
        */
        //canvas.restore();

    }

}
