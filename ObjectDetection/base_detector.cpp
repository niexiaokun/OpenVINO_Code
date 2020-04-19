//
// Created by kun on 2020/4/17.
//

#include "base_detector.h"
#include <opencv2/opencv.hpp>

base_detector::base_detector():conf_th(0.5) {

}

base_detector::~base_detector() {

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