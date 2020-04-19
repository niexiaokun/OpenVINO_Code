//
// Created by kun on 2020/4/17.
//

#include "yolo_detector.h"
#include "helper.h"
#include "slog.hpp"
#include "common.hpp"
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>
#ifdef WITH_EXTENSIONS
#include <ext_list.hpp>
#endif

using namespace InferenceEngine;

yolo_detector::yolo_detector() {

}

yolo_detector::~yolo_detector() {

}

bool yolo_detector::initiate(InferenceEngine::Core &ie, const std::string &model_path, const std::string &IE_device) {
    slog::info << "Loading Inference Engine" << slog::endl;

#ifdef WITH_EXTENSIONS
    if (IE_device == "CPU") {
        ie.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>(), "CPU");
    }
#endif
    slog::info << "Device info" << slog::endl;
    std::cout << ie.GetVersions(IE_device);
    slog::info << "Loading network files" << slog::endl;

    this->networkReader.ReadNetwork(model_path);
    std::string binFileName = fileNameNoExt(model_path) + ".bin";
    this->networkReader.ReadWeights(binFileName);
    std::string labelFileName = fileNameNoExt(model_path) + ".labels";
    std::ifstream inputFile(labelFileName);
    std::copy(std::istream_iterator<std::string>(inputFile),
              std::istream_iterator<std::string>(),
              std::back_inserter(this->labels));
    this->network = networkReader.getNetwork();
    this->network.setBatchSize(1);
    slog::info << "Batch size is " << std::to_string(networkReader.getNetwork().getBatchSize()) << slog::endl;

    slog::info << "Preparing input blobs" << slog::endl;
    InputsDataMap inputInfo(this->network.getInputsInfo());
    this->InputName = inputInfo.begin()->first;
    InputInfo::Ptr& inputData = inputInfo.begin()->second;
    inputData->setPrecision(Precision::U8);
    //size_t num_channels = inputData->getTensorDesc().getDims()[1];
    this->input_height = inputData->getTensorDesc().getDims()[2];
    this->input_width = inputData->getTensorDesc().getDims()[3];
    this->InputSize = cv::Size(this->input_width, this->input_height);
    slog::info << "model input size is  " <<
               std::to_string(this->input_height)+" x "+std::to_string(this->input_width) << slog::endl;
#if 0
    inputData->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
            inputData->getInputData()->setLayout(Layout::NHWC);
#else
    inputData->getInputData()->setLayout(Layout::NCHW);
#endif

    slog::info << "Preparing output blobs" << slog::endl;
    this->outputInfo = this->network.getOutputsInfo();
    OutputName = outputInfo.begin()->first;
    for_each(outputInfo.begin(), outputInfo.end(), [&](const std::pair<std::string, DataPtr>& item){
        this->OutputName = item.first;
        DataPtr outputData = item.second;
        outputData->setPrecision(Precision::FP32);
        auto out_layer = this->network.getLayerByName(this->OutputName.c_str());
        assert(out_layer->type == "RegionYolo");
    });

    slog::info << "Loading model to the device" << slog::endl;
    this->executable_network = ie.LoadNetwork(network, IE_device, {});
    slog::info << "Create infer request" << slog::endl;
    this->curr_infer_request = this->executable_network.CreateInferRequestPtr();
    this->next_infer_request = this->executable_network.CreateInferRequestPtr();

}


bool yolo_detector::start_async_exec(cv::Mat &curr_frame, cv::Mat &next_frame, cv::Mat &result) {
    if(isFirstFrame){
        helper::frameToBlob(curr_frame, curr_infer_request, InputName);
        curr_infer_request->StartAsync();
        isFirstFrame = false;
    } else{
        helper::frameToBlob(next_frame, next_infer_request, InputName);
        next_infer_request->StartAsync();
    }

    if (OK == curr_infer_request->Wait(IInferRequest::WaitMode::RESULT_READY)){
        std::vector<DetectionObject> objects;
        // Parsing outputs
        for (auto &output : this->outputInfo) {
            auto output_name = output.first;
            auto out_layer = network.getLayerByName(output_name.c_str());
            Blob::Ptr blob = curr_infer_request->GetBlob(output_name);
            ParseYOLOV3Output(out_layer, blob, objects);
        }
        // Filtering overlapping boxes
        auto objSize = objects.size();
        std::sort(objects.begin(), objects.end(), std::greater<DetectionObject>());
        for (size_t i = 0; i < objects.size(); ++i) {
            if (objects[i].confidence == 0)
                continue;
            for (size_t j = i + 1; j < objects.size(); ++j)
                if (objects[i].IntersectionOverUnion(objects[j]) >= iou_t)
                    objects[j].confidence = 0;
        }
        this->draw_detection_result(result, objects);
    }
    curr_infer_request.swap(next_infer_request);
    //next_frame.copyTo(curr_frame);
    return true;
}

void yolo_detector::ParseYOLOV3Output(const CNNLayerPtr &layer, const Blob::Ptr &blob,
                                      std::vector<DetectionObject> &objects) {
    const int out_blob_h = static_cast<int>(blob->getTensorDesc().getDims()[2]);
    const int out_blob_w = static_cast<int>(blob->getTensorDesc().getDims()[3]);
    auto num = layer->GetParamAsInt("num");
    auto coords = layer->GetParamAsInt("coords");
    auto classes = layer->GetParamAsInt("classes");
    auto anchors = layer->GetParamAsFloats("anchors");
    auto mask = layer->GetParamAsInts("mask");
    num = mask.size();
    std::vector<float> maskedAnchors(num * 2);
    for (int i = 0; i < num; ++i) {
        maskedAnchors[i * 2] = anchors[mask[i] * 2];
        maskedAnchors[i * 2 + 1] = anchors[mask[i] * 2 + 1];
    }
    anchors = maskedAnchors;

    auto side = out_blob_h;
    auto side_square = side * side;
    const float *output_blob = blob->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
    for(int i=0; i<side_square; i++){
        int row = i / side;
        int col = i % side;
        for(int n=0; n<num; n++){
            int box_index = n * side_square * (coords + classes +1) + i;
            int obj_index = n * side_square * (coords + classes +1) + coords * side_square + i;
            float scale = output_blob[obj_index];
            if (scale < conf_th)
                continue;
            double x = (col + output_blob[box_index + 0 * side_square]) / side * this->input_width;
            double y = (row + output_blob[box_index + 1 * side_square]) / side * this->input_height;
            double height = std::exp(output_blob[box_index + 3 * side_square]) * anchors[2 * n + 1];
            double width = std::exp(output_blob[box_index + 2 * side_square]) * anchors[2 * n];
            for(int j=0; j<classes; j++){
                int class_index = n * side_square * (coords + classes +1) + (coords + 1 + j) * side_square + i;
                float prob = scale * output_blob[class_index];
                if (prob < conf_th)
                    continue;
                DetectionObject obj(x, y, height, width, j, prob, 1, 1);
                objects.push_back(obj);
            }
        }
    }
}
