#ifndef MTCNN_DETECT_H
#define MTCNN_DETECT_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <vector>

static const float pnet_th = 0.7f;
static const float rnet_th = 0.7f;
static const float onet_th = 0.2f;
static const float min_objsize = 24.0f;
static const float pyramid_factor = 0.666f;
static const float max_objsize = 70.0f;
static const int max_pnet_bbox_num = 100; // 100
static const int max_rnet_bbox_num = 50; // 50
static const float pnet_merge_th = 0.6f;
static const float rnet_merge_th = 0.5f;

typedef struct
    {
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    float score;
    } obj_box;

typedef struct
    {
    float bbox_reg[4];
    obj_box bbox;
    } obj_info;

void init_mtcnn
    (
    const int srcw,
    const int srch
    );

void run_mtcnn
    (
    cv::Mat& im,
    std::vector<obj_info>& onet_boxes
    );

#endif // MTCNN_DETECT_H
