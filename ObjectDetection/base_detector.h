//
// Created by kun on 2020/4/17.
//

#ifndef OPENVINO_CODE_BASE_DETECTOR_H
#define OPENVINO_CODE_BASE_DETECTOR_H

#include <iostream>
#include <inference_engine.hpp>
#include <opencv2/core/types.hpp>
#include <chrono>

using namespace InferenceEngine;

struct DetectionObject {
    int xmin, ymin, xmax, ymax, class_id;
    float confidence;

    DetectionObject(double x, double y, double h, double w, int class_id, float confidence, float h_scale, float w_scale) {
        this->xmin = static_cast<int>((x - w / 2) * w_scale);
        this->ymin = static_cast<int>((y - h / 2) * h_scale);
        this->xmax = static_cast<int>(this->xmin + w * w_scale);
        this->ymax = static_cast<int>(this->ymin + h * h_scale);
        this->class_id = class_id;
        this->confidence = confidence;
    }

    DetectionObject(float xmin, float ymin, float xmax, float ymax, float confidence, int class_id){
        this->xmin = static_cast<int>(xmin);
        this->ymin = static_cast<int>(ymin);
        this->xmax = static_cast<int>(xmax);
        this->ymax = static_cast<int>(ymax);
        this->class_id = class_id;
        this->confidence = confidence;
    }

    double IntersectionOverUnion(const DetectionObject &obj){
        double width_of_overlap_area = fmin(this->xmax, obj.xmax) - fmax(this->xmin, obj.xmin);
        double height_of_overlap_area = fmin(this->ymax, obj.ymax) - fmax(this->ymin, obj.ymin);
        double area_of_overlap;
        if (width_of_overlap_area < 0 || height_of_overlap_area < 0)
            area_of_overlap = 0;
        else
            area_of_overlap = width_of_overlap_area * height_of_overlap_area;
        double box_1_area = (this->ymax - this->ymin)  * (this->xmax - this->xmin);
        double obj_area = (obj.ymax - obj.ymin)  * (obj.xmax - obj.xmin);
        double area_of_union = box_1_area + obj_area - area_of_overlap;
        return area_of_overlap / area_of_union;
    }

    bool operator <(const DetectionObject &s2) const {
        return this->confidence < s2.confidence;
    }
    bool operator >(const DetectionObject &s2) const {
        return this->confidence > s2.confidence;
    }
};

class base_detector {
public:
    base_detector();
    ~base_detector();

public:
    cv::Size InputSize;

protected:
    InferenceEngine::CNNNetReader networkReader;
    InferenceEngine::CNNNetwork network;
    InferenceEngine::ExecutableNetwork executable_network;
    InferenceEngine::OutputsDataMap outputInfo;

    InferenceEngine::InferRequest::Ptr curr_infer_request;
    InferenceEngine::InferRequest::Ptr next_infer_request;

    size_t num_classes;
    size_t input_height;
    size_t input_width;
    cv::Size OutputSize;

    size_t maxProposal;
    size_t objectSize;

    std::string InputName;
    std::string OutputName;
    std::vector<std::string> labels;

    float iou_t;
    float conf_th;
    bool isFirstFrame;
    
    void draw_detection_result(cv::Mat& image, std::vector<DetectionObject>& detectObjs);
};


#endif //OPENVINO_CODE_BASE_DETECTOR_H
