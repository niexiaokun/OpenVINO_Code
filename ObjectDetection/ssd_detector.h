//
// Created by kun on 2020/4/17.
//

#ifndef OPENVINO_CODE_SSD_DETECTOR_H
#define OPENVINO_CODE_SSD_DETECTOR_H


#include "base_detector.h"

class ssd_detector : public base_detector{
public:
    ssd_detector();
    ~ssd_detector();

    bool initiate(InferenceEngine::Core& ie, const std::string& model_path, const std::string& IE_device);
    bool start_async_exec(cv::Mat& curr_frame, cv::Mat& next_frame, cv::Mat& result);

private:
    void postprocess(const float* detections, std::vector<DetectionObject>& detectionObjects);

};


#endif //OPENVINO_CODE_SSD_DETECTOR_H
