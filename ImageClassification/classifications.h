//
// Created by kun on 2020/5/30.
//

#ifndef OPENVINO_CODE_CLASSIFICATIONS_H
#define OPENVINO_CODE_CLASSIFICATIONS_H

#include <iostream>
#include <inference_engine.hpp>
#include <opencv2/opencv.hpp>

class classifications {

public:
    classifications();
    ~classifications();

    bool initiate(InferenceEngine::Core& ie, const std::string& model_path, const std::string& IE_device);
    bool start_async_exec(cv::Mat& curr_frame, cv::Mat& next_frame, cv::Mat& result);
    bool start_sync_exec(cv::Mat& curr_frame, cv::Mat& result);

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
    cv::Size InputSize;
    cv::Size OutputSize;

    std::string InputName;
    std::string OutputName;
    std::vector<std::string> labels;

    bool isFirstFrame;

};


#endif //OPENVINO_CODE_CLASSIFICATIONS_H
