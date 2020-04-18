//
// Created by kun on 2020/4/17.
//

#include "base_detector.h"
#include <opencv2/opencv.hpp>

base_detector::base_detector():conf_th(0.5) {

}

base_detector::~base_detector() {

}

void base_detector::frameToBlob(cv::Mat &image, InferRequest::Ptr &inferRequest, std::string &input_name) {
    Blob::Ptr input_blob = inferRequest->GetBlob(input_name);
    InferenceEngine::SizeVector blobSize = input_blob->getTensorDesc().getDims();
    const size_t width = blobSize[3];
    const size_t height = blobSize[2];
    const size_t channels = blobSize[1];
    auto input_buffer = input_blob->buffer().as<PrecisionTrait<Precision::U8>::value_type*>();
    for (size_t ch = 0; ch < channels; ++ch){
        for (size_t  h = 0; h < height; h++){
            uchar* pResizeImage;
            pResizeImage = image.ptr<uchar>(h);
            for (size_t w = 0; w < width; w++){
                input_buffer[ch * width * height + h * width + w] = *(pResizeImage+3*w+ch);
            }
        }
    }
}

void base_detector::draw_detection_result(cv::Mat &image, std::vector<DetectionObject> &detectObjs) {
    for_each(detectObjs.begin(), detectObjs.end(), [&](const DetectionObject& detectObj){
        if(detectObj.confidence > conf_th){
            cv::Point pt1(detectObj.xmin, detectObj.ymin);
            cv::Point pt2(detectObj.xmax, detectObj.ymax);
            cv::rectangle(image, pt1, pt2, cv::Scalar(0,0,255), -1);
        }
    });
}