//
// Created by kun on 2020/4/18.
//

#include "maskRCNN.h"
#include <iostream>
#include <chrono>
#include "slog.hpp"
#include "ocv_common.hpp"
#include <opencv2/opencv.hpp>

#define ASYNC_EXEC

typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;


int main(int argc, char *argv[]){

    InferenceEngine::Core ie;
    maskRCNN maskRcnn;
    std::string model_path = "../../ir_models/mask_rcnn_R_50_FPN_1x/mask_rcnn_R_50_FPN_1x.xml";
    maskRcnn.initiate(ie, model_path, "CPU");

    std::string video_path = "/home/kun/dataset/videos/person.mp4";
    cv::VideoCapture video_capture(video_path);
    size_t wait_time = 1;
    cv::Mat curr_frame;
    cv::Mat next_frame;
    cv::Mat result;
    cv::Mat out_frame;

    bool isFirstFrame = true;

    slog::info << "Start inference" << slog::endl;
    while(video_capture.isOpened()){
        auto t0 = std::chrono::high_resolution_clock::now();
#ifdef ASYNC_EXEC
        if(isFirstFrame){
            video_capture >> curr_frame;
            if(curr_frame.empty()){
                slog::info << "first frame is empty" << slog::endl;
                return -1;
            }
            isFirstFrame = false;
//                cv::resize(curr_frame, curr_frame, detectionModel.InputSize);
            cv::resize(curr_frame, curr_frame, maskRcnn.InputSize);
        }
        video_capture >> next_frame;
        if(next_frame.empty()){
            slog::info << "frame is empty" << slog::endl;
            break;
        }
        out_frame = curr_frame.clone();
        result = curr_frame.clone();
        cv::resize(next_frame, next_frame, maskRcnn.InputSize);
        maskRcnn.start_async_exec(curr_frame, next_frame, result);
        next_frame.copyTo(curr_frame);
#else
        video_capture >> curr_frame;
            if(curr_frame.empty()){
                slog::info << "first frame is empty" << slog::endl;
                return -1;
            }
            cv::resize(curr_frame, curr_frame, maskRcnn.InputSize);
            out_frame = curr_frame.clone();
            result = curr_frame.clone();
            maskRcnn.start_sync_exec(curr_frame, result);
#endif

        auto t1 = std::chrono::high_resolution_clock::now();
        ms det_time = std::chrono::duration_cast<ms>(t1 - t0);
        std::ostringstream ss;
        ss << "detection time : " << std::fixed << std::setprecision(2) << det_time.count() << " ms";
        std::cout << ss.str() << std::endl;
        putText(result, ss.str(), cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2, 8);
        cv::imshow("result", result);

        int key = cv::waitKey(wait_time);

        if (27 == key){
            video_capture.release();
            break;
        }else if(32 == key){
            wait_time = !wait_time;
        }
    }
    cv::destroyAllWindows();

    return 0;
}