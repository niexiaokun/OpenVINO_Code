//
// Created by kun on 2020/4/18.
//

#ifndef OPENVINO_CODE_MASKRCNN_H
#define OPENVINO_CODE_MASKRCNN_H

#include <iostream>
#include <inference_engine.hpp>
#ifdef WITH_EXTENSIONS
#include <ext_list.hpp>
#endif
#include <opencv2/core/types.hpp>
#include <chrono>

using namespace InferenceEngine;

class maskRCNN {
public:
    maskRCNN();
    ~maskRCNN();

    bool initiate(InferenceEngine::Core& ie, const std::string& model_path, const std::string& IE_device);
    bool start_async_exec(cv::Mat& curr_frame, cv::Mat& next_frame, cv::Mat& result);

public:
    cv::Size InputSize;

private:
    InferenceEngine::CNNNetReader networkReader;
    InferenceEngine::CNNNetwork network;
    InferenceEngine::ExecutableNetwork executable_network;
    InferenceEngine::InputsDataMap  inputInfo;
    InferenceEngine::OutputsDataMap outputInfo;

    InferenceEngine::InferRequest::Ptr curr_infer_request;
    InferenceEngine::InferRequest::Ptr next_infer_request;

    size_t input_height;
    size_t input_width;
    size_t num_classes;
    cv::Size OutputSize;

    std::string InputName;
    std::string OutputName;
    std::vector<std::string> labels;

    bool isFirstFrame;

    void frameToBlob(cv::Mat &image, InferenceEngine::InferRequest::Ptr &infer_request, InputsDataMap &inputInfo);

};


#endif //OPENVINO_CODE_MASKRCNN_H
