#include <jni.h>
#include <string>
#include <algorithm>
#define PROTOBUF_USE_DLLS 1
#define CAFFE2_USE_LITE_PROTO 1
#include <caffe2/predictor/predictor.h>
#include <caffe2/core/operator.h>
#include <caffe2/core/timer.h>

#include "caffe2/core/init.h"

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>

#include "classes.h"
#include "libyuv.h"

#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xsort.hpp>

#define IMG_H 448
#define IMG_W 448
#define IMG_C 3
#define MAX_DATA_SIZE IMG_H * IMG_W * IMG_C
#define alog(...) __android_log_print(ANDROID_LOG_ERROR, "AICamera", __VA_ARGS__);

static caffe2::NetDef _initNet, _predictNet;
static caffe2::Predictor *_predictor;
static float input_data[MAX_DATA_SIZE];
static caffe2::Workspace ws;
const char * classes[] {
"aeroplane",
"bicycle",
"bird",
"boat",
"bottle",
"bus",
"car",
"cat",
"chair",
"cow",
"diningtable",
"dog",
"horse",
"motorbike",
"person",
"pottedplant",
"sheep",
"sofa",
"train",
"tvmonitor"};



struct Bbox {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int index;
    Bbox(float x1_, float y1_, float x2_, float y2_, float s, int index_):
        x1(x1_), y1(y1_), x2(x2_), y2(y2_), score(s), index(index_) {};
};

float iou(Bbox box1, Bbox box2) {
    float area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
    float area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
    float x11 = std::max(box1.x1, box2.x1);
    float y11 = std::max(box1.y1, box2.y1);
    float x22 = std::min(box1.x2, box2.x2);
    float y22 = std::min(box1.y2, box2.y2);
    float intersection = (x22 - x11) * (y22 - y11);
    return intersection / (area1 + area2 - intersection);
}

std::vector<Bbox> nms(std::vector<Bbox> &vecBbox, float threshold) {
    auto cmpScore = [](Bbox box1, Bbox box2) {
        return box1.score < box2.score; // 升序排列, 令score最大的box在vector末端
    };
    std::sort(vecBbox.begin(), vecBbox.end(), cmpScore);

    std::vector<Bbox> pickedBbox;
    while (vecBbox.size() > 0) {
        pickedBbox.emplace_back(vecBbox.back());
        vecBbox.pop_back();
        for (size_t i = 0; i < vecBbox.size(); i++) {
            if (iou(pickedBbox.back(), vecBbox[i]) >= threshold) {
                vecBbox.erase(vecBbox.begin() + i);
            }
        }
    }
    return pickedBbox;
}


using namespace xt::placeholders;  // required for `_` to work

// A function to load the NetDefs from protobufs.
void loadToNetDef(AAssetManager* mgr, caffe2::NetDef* net, const char *filename) {
    AAsset* asset = AAssetManager_open(mgr, filename, AASSET_MODE_BUFFER);
    assert(asset != nullptr);
    const void *data = AAsset_getBuffer(asset);
    assert(data != nullptr);
    off_t len = AAsset_getLength(asset);
    assert(len != 0);
    if (!net->ParseFromArray(data, len)) {
        alog("Couldn't parse net from data.\n");
    }
    AAsset_close(asset);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_ufo_aicamera_MainActivity_initCaffe2(JNIEnv *env, jobject /* this */, jobject assetManager) {

    AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
    alog("Attempting to load protobuf netdefs...");
    loadToNetDef(mgr, &_initNet,   "yolo_init_net.pb");
    loadToNetDef(mgr, &_predictNet,"yolo_predict_net.pb");
    alog("done.");
    alog("Instantiating predictor...");
    _predictor = new caffe2::Predictor(_initNet, _predictNet);
    alog("done.")

}

float avg_fps = 0.0;
float total_fps = 0.0;
int iters_fps = 10;

extern "C"
JNIEXPORT jstring JNICALL
Java_com_ufo_aicamera_MainActivity_detectionFromCaffe2(
        JNIEnv *env,
        jobject /* this */,
        jint h, jint w, jbyteArray Y, jbyteArray U, jbyteArray V,
        jint rowStride, jint pixelStride,
        jboolean infer_HWC) {



    jsize Y_len = env->GetArrayLength(Y);
    jbyte * Y_data = env->GetByteArrayElements(Y, 0);
    assert(Y_len <= MAX_DATA_SIZE);
    jsize U_len = env->GetArrayLength(U);
    jbyte * U_data = env->GetByteArrayElements(U, 0);
    assert(U_len <= MAX_DATA_SIZE);
    jsize V_len = env->GetArrayLength(V);
    jbyte * V_data = env->GetByteArrayElements(V, 0);
    assert(V_len <= MAX_DATA_SIZE);

#define min(a,b) ((a) > (b)) ? (b) : (a)
#define max(a,b) ((a) > (b)) ? (a) : (b)

    auto h_offset = max(0, (h - IMG_H) / 2);
    auto w_offset = max(0, (w - IMG_W) / 2);

    auto iter_h = IMG_H;
    auto iter_w = IMG_W;
    if (h < IMG_H) {
        iter_h = h;
    }
    if (w < IMG_W) {
        iter_w = w;
    }

    for (auto i = 0; i < iter_h; ++i) {
        jbyte* Y_row = &Y_data[(h_offset + i) * w];
        jbyte* U_row = &U_data[(h_offset + i) / 2 * rowStride];
        jbyte* V_row = &V_data[(h_offset + i) / 2 * rowStride];
        for (auto j = 0; j < iter_w; ++j) {
            // Tested on Pixel and S7.
            char y = Y_row[w_offset + j];
            char u = U_row[pixelStride * ((w_offset+j)/pixelStride)];
            char v = V_row[pixelStride * ((w_offset+j)/pixelStride)];

            float b_mean = 104.00698793f;
            float g_mean = 116.66876762f;
            float r_mean = 122.67891434f;

            auto b_i = 0 * IMG_H * IMG_W + j * IMG_W + i;
            auto g_i = 1 * IMG_H * IMG_W + j * IMG_W + i;
            auto r_i = 2 * IMG_H * IMG_W + j * IMG_W + i;

            if (infer_HWC) {
                b_i = (j * IMG_W + i) * IMG_C;
                g_i = (j * IMG_W + i) * IMG_C + 1;
                r_i = (j * IMG_W + i) * IMG_C + 2;
            }
/*
  R = Y + 1.402 (V-128)
  G = Y - 0.34414 (U-128) - 0.71414 (V-128)
  B = Y + 1.772 (U-V)
 */
            input_data[r_i] = -r_mean + (float) ((float) min(255., max(0., (float) (y + 1.402 * (v - 128)))));
            input_data[g_i] = -g_mean + (float) ((float) min(255., max(0., (float) (y - 0.34414 * (u - 128) - 0.71414 * (v - 128)))));
            input_data[b_i] = -b_mean + (float) ((float) min(255., max(0., (float) (y + 1.772 * (u - v)))));

        }
    }


    caffe2::TensorCPU input = caffe2::Tensor(1,caffe2::DeviceType::CPU);

    input.Resize(std::vector<int>({1, IMG_C, IMG_H, IMG_W}));


    memcpy(input.mutable_data<float>(), input_data, IMG_H * IMG_W * IMG_C * sizeof(float));
    caffe2::Predictor::TensorList input_vec{input};
    caffe2::Predictor::TensorList output_vec;
    caffe2::Timer t;
    t.Start();
    _predictor->operator()(input_vec, &output_vec);
    float fps = 1000/t.MilliSeconds();
    total_fps += fps;
    avg_fps = total_fps / iters_fps;
    total_fps -= avg_fps;

    int grid_num = 14;
    auto cell_size = 1.0 / grid_num;
    //std::vector<xt::xarray<int> > boxes;
    //std::vector<xt::xarray<float> > cls_indexes;
    //std::vector<xt::xarray<float> > probs;
    std::vector<float> output;
    std::vector<std::vector<std::vector<float>>> boxes;
    for (auto out : output_vec){
        int count = 0;
        for (int j = 0; j < 14; j ++){
            std::vector<std::vector<float>> temp_1;
            for( int k = 0; k < 14; k ++){
                std::vector<float> temp_2;
                for(int l = 0; l < 30; l ++){
                    temp_2.push_back(out.template data<float>()[count]);
                    count ++;
                }
                temp_1.push_back(temp_2);
            }
            boxes.push_back(temp_1);
        }
    }

    std::vector<std::vector<std::vector<float>>> contain (14,std::vector<std::vector<float>>(14,std::vector<float>(2)));
    std::vector<std::vector<std::vector<int>>> mask1 (14,std::vector<std::vector<int>>(14,std::vector<int>(2)));

    float max_ = boxes[0][0][4];
    for(int i = 0; i < 14; i ++){
        for (int j = 0; j < 14; j ++){
            contain[i][j][0] = boxes[i][j][4];
            mask1[i][j][0] = contain[i][j][0] > 0.2 ? 1 : 0;
            contain[i][j][1] = boxes[i][j][9];
            mask1[i][j][1] = contain[i][j][1] > 0.2 ? 1 : 0;
            if(boxes[i][j][4] > max_)
                max_ = boxes[i][j][4];
            if(boxes[i][j][9] > max_)
                max_ = boxes[i][j][9];
        }
    }
    std::vector<std::vector<std::vector<int>>> mask2 (14,std::vector<std::vector<int>>(14,std::vector<int>(2)));
    for(int i = 0; i < 14; i ++){
        for (int j = 0; j < 14; j ++){
            mask2[i][j][0] = contain[i][j][0] == max_ ? 1: 0;
            mask2[i][j][0] += mask1[i][j][0];
            mask2[i][j][1] = contain[i][j][1] == max_ ? 1: 0;
            mask2[i][j][1] += mask1[i][j][1];
        }
    }

    std::vector<Bbox> final_box;

    for(int i = 0 ; i < grid_num; i ++){
        for (int j = 0; j < grid_num; j ++){
            for(int b = 0; b < 2; b ++){
                if(mask2[i][j][b] > 0){
                    float contain_prob = boxes[i][j][b*5+4];
                    std::vector<float> box(4);
                    for(int p = 0; p < 4; p ++)
                        box[p] = boxes[i][j][b*5+p];
                    box[0] = (box[0] + j) * cell_size;
                    box[1] = (box[1] + i) * cell_size;
                    std::vector<float> box_xy(4);
                    box_xy[0] = box[0] - 0.5 * box[2];
                    box_xy[1] = box[1] - 0.5 * box[3];
                    box_xy[2] = box[0] + 0.5 * box[2];
                    box_xy[3] = box[1] + 0.5 * box[3];
                    float prob_max = boxes[i][j][10];
                    int index =0;
                    for(int p =10; p < 30; p ++){
                        if (boxes[i][j][p] > prob_max){
                            prob_max = boxes[i][j][p];
                            index = p - 10;
                        }
                    }
                    if(contain_prob * prob_max > 0.2){
                        Bbox temp = Bbox(box_xy[0]*448, box_xy[1]*448, box_xy[2]*448, box_xy[3]*448, contain_prob * prob_max, index);
                        final_box.push_back(temp);
                    }
                }
            }
        }
    }

    std::vector<Bbox> answer = nms(final_box, 0.25);
    std::ostringstream stringStream;
    stringStream << "/";
    std::vector<float> res;
    for (const Bbox& p : answer){
        stringStream << p.x1 << "+";
        stringStream << p.y1 << "+";
        stringStream << p.x2 << "+";
        stringStream << p.y2 << "+";
        stringStream << p.score << "+";
        stringStream << p.index << "/";
    }

    constexpr int k = 5;
    float max[k] = {0};
    int max_index[k] = {0};
    // Find the top-k results manually.
    /*
    if (output_vec.capacity() > 0) {
        for (auto output : output_vec) {
            for (auto i = 0; i < output.size(); ++i) {
                for (auto j = 0; j < k; ++j) {
                    if (output.template data<float>()[i] > max[j]) {
                        for (auto _j = k - 1; _j > j; --_j) {
                            max[_j - 1] = max[_j];
                            max_index[_j - 1] = max_index[_j];
                        }
                        max[j] = output.template data<float>()[i];
                        max_index[j] = i;
                        goto skip;
                    }
                }
                skip:;
            }
        }
    }
     */


    return env->NewStringUTF(stringStream.str().c_str());

    //std::ostringstream stringStream;
    //stringStream << avg_fps << " FPS\n";

    /*
    for (auto j = 0; j < answer.size(); ++j) {
        stringStream << classes[answer[j].index] << "\n";
    }

    return env->NewStringUTF(stringStream.str().c_str());
     */
}



