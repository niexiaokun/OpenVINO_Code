//
// Created by kun on 2020/4/17.
//

#ifndef OPENVINO_CODE_YOLO_DETECTOR_H
#define OPENVINO_CODE_YOLO_DETECTOR_H


#include "base_detector.h"

class yolo_detector: public base_detector {
public:
    yolo_detector();
    ~yolo_detector();

    bool initiate(InferenceEngine::Core& ie, const std::string& model_path, const std::string& IE_device);
    bool start_async_exec(cv::Mat& curr_frame, cv::Mat& next_frame, cv::Mat& result);

private:
    void ParseYOLOV3Output(const CNNLayerPtr &layer, const Blob::Ptr &blob, std::vector<DetectionObject> &objects);
};


#endif //OPENVINO_CODE_YOLO_DETECTOR_H
