//
// Created by kun on 2020/4/19.
//

#ifndef OPENVINO_CODE_HELPER_H
#define OPENVINO_CODE_HELPER_H

#endif //OPENVINO_CODE_HELPER_H

#include <opencv2/opencv.hpp>

namespace helper{

    void frameToBlob(cv::Mat &image, InferRequest::Ptr &inferRequest, std::string &input_name) {
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
}