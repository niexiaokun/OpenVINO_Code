//
// Created by kun on 2020/4/17.
//

#include <iostream>
#include "ssd_detector.h"
#include "slog.hpp"
#include "common.hpp"
#include <opencv2/opencv.hpp>

#define ASYNC_EXEC

int main(int argc, char *argv[]){

    try {
        std::string video_path;
        std::string model_path = "../../ir_models/ssd_v1_coco/frozen_inference_graph.xml";
        InferenceEngine::Core ie;
        ssd_detector detector;

        detector.initiate(ie, model_path, "CPU");

        video_path = "/home/kun/dataset/videos/car.mp4";
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
                cv::resize(curr_frame, curr_frame, detector.InputSize);
//                cv::resize(curr_frame, curr_frame, multi_detectModel.InputSizes[0]);
            }
            video_capture >> next_frame;
            if(next_frame.empty()){
                slog::info << "frame is empty" << slog::endl;
                break;
            }
            out_frame = curr_frame.clone();
            result = curr_frame.clone();
            cv::resize(next_frame, next_frame, detector.InputSize);
            detector.start_async_exec(curr_frame, next_frame, result);
            next_frame.copyTo(curr_frame);
#else
            video_capture >> curr_frame;
            if(curr_frame.empty()){
                slog::info << "first frame is empty" << slog::endl;
                return -1;
            }
            cv::resize(curr_frame, curr_frame, detector.InputSize);
            out_frame = curr_frame.clone();
            result = curr_frame.clone();
            detector.start_sync_exec(curr_frame, result);
#endif

            cv::addWeighted(out_frame, 0.7, result, 0.3, 0, out_frame, CV_8UC3);

            auto t1 = std::chrono::high_resolution_clock::now();
            ms det_time = std::chrono::duration_cast<ms>(t1 - t0);
            std::ostringstream ss;
            ss << "detection time : " << std::fixed << std::setprecision(2) << det_time.count() << " ms";
            std::cout << ss.str() << std::endl;
            putText(out_frame, ss.str(), cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2, 8);
            cv::hconcat(curr_frame, out_frame, out_frame);
            cv::imshow("display", out_frame);
//            cv::imshow("result", result);

            int key = cv::waitKey(wait_time);

            if (27 == key){
                video_capture.release();
                break;
            }else if(32 == key){
                wait_time = !wait_time;
            }
        }
        cv::destroyAllWindows();

    }
    catch (const std::exception& error) {
        slog::err << error.what() << slog::endl;
        return 1;
    }
    catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }
    slog::info << "Execution successful" << slog::endl;

    return 0;
}