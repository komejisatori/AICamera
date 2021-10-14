package com.ufo.aicamera;


import android.graphics.RectF;

public class Recognition {
    public RectF bbox;
    public String cls;
    public Float prob;

    private String [] class_names = new String[]{"aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
            "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
            "pottedplant", "sheep", "sofa", "train", "tvmonitor"};

    Recognition(RectF box, Integer _cls_idx, Float _prob){
        this.bbox = box;
        this.cls = class_names[_cls_idx];
        this.prob = _prob;
    }
}