//
// Created by kun on 2020/5/30.
//

#include <chrono>
#include "slog.hpp"
#include "classifications.h"

#define ASYNC_EXEC

typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;

int main(int argc, char *argv[]){

    try {
        std::string video_path;
        std::string model_path = "../../ir_models/ssd_v1_coco/frozen_inference_graph.xml";
        InferenceEngine::Core ie;
        classifications image_classifier;
        image_classifier.initiate(ie, model_path, "CPU");

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