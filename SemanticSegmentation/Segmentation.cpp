//
// Created by kun on 2020/4/19.
//

#include "Segmentation.h"
#include "helper.h"
#include "common.hpp"
#include "slog.hpp"

typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;

Segmentation::Segmentation() {

}

Segmentation::~Segmentation() {

}

bool Segmentation::initiate(InferenceEngine::Core &ie, const std::string &model_path, const std::string &IE_device) {
    slog::info << "Loading Inference Engine" << slog::endl;

#ifdef WITH_EXTENSIONS
    if (IE_device == "CPU") {
        ie.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>(), "CPU");
    }
#endif
    slog::info << "Device info" << slog::endl;
    std::cout << ie.GetVersions(IE_device);
    slog::info << "Loading network files" << slog::endl;

    this->network = ie.ReadNetwork(model_path);
    this->network.setBatchSize(1);
    slog::info << "Batch size is " << std::to_string(this->network.getBatchSize()) << slog::endl;

    slog::info << "Preparing input blobs" << slog::endl;
    this->inputInfo = network.getInputsInfo();
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

    const OutputsDataMap& outputsDataMap = network.getOutputsInfo();
    if (outputsDataMap.size() != 1) throw std::runtime_error("Demo supports topologies only with 1 output");
    OutputName = outputsDataMap.begin()->first;
    Data& data = *outputsDataMap.begin()->second;
    // if the model performs ArgMax, its output type can be I32 but for models that return heatmaps for each
    // class the output is usually FP32. Reset the precision to avoid handling different types with switch in
    // postprocessing
    data.setPrecision(Precision::FP32);
    const SizeVector& outSizeVector = data.getTensorDesc().getDims();
    switch(outSizeVector.size()) {
        case 3:
            outChannels = 0;
            outHeight = outSizeVector[1];
            outWidth = outSizeVector[2];
            break;
        case 4:
            outChannels = outSizeVector[1]==1? 0 : outSizeVector[1];
            outHeight = outSizeVector[2];
            outWidth = outSizeVector[3];
            break;
        default:
            throw std::runtime_error("Unexpected output blob shape. Only 4D and 3D output blobs are"
                                     "supported.");
    }

    slog::info << "Loading model to the device" << slog::endl;
    executable_network = ie.LoadNetwork(network, IE_device, {});
    slog::info << "Create infer request" << slog::endl;
    curr_infer_request = executable_network.CreateInferRequestPtr();
    next_infer_request = executable_network.CreateInferRequestPtr();

}

bool Segmentation::start_async_exec(cv::Mat &curr_frame, cv::Mat &next_frame, cv::Mat &resImg) {

    if(isFirstFrame){
        helper::frameToBlob(curr_frame, curr_infer_request, this->InputName);
        curr_infer_request->StartAsync();
        isFirstFrame = false;
    } else{
        helper::frameToBlob(next_frame, next_infer_request, this->InputName);
        next_infer_request->StartAsync();
    }

    if (OK == curr_infer_request->Wait(IInferRequest::WaitMode::RESULT_READY)) {
        const Blob::Ptr output_blob = curr_infer_request->GetBlob(OutputName);

        cv::Mat maskImg = cv::Mat::zeros(outHeight, outWidth, CV_8UC3);
        std::vector<cv::Vec3b> colors(arraySize(CITYSCAPES_COLORS));
        for (std::size_t i = 0; i < colors.size(); ++i)
            colors[i] = {CITYSCAPES_COLORS[i].blue(), CITYSCAPES_COLORS[i].green(), CITYSCAPES_COLORS[i].red()};
        std::mt19937 rng;
        std::uniform_int_distribution<int> distr(0, 255);

        std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();

        const float *const predictions = curr_infer_request->GetBlob(OutputName)->cbuffer().as<float *>();
        for (int rowId = 0; rowId < outHeight; ++rowId) {
            for (int colId = 0; colId < outWidth; ++colId) {
                std::size_t classId = 0;
                if (outChannels == 0) {  // assume the output is already ArgMax'ed
                    classId = static_cast<std::size_t>(predictions[rowId * outWidth + colId]);
                } else {
                    float maxProb = -1.0f;
                    for (int chId = 0; chId < outChannels; ++chId) {
                        float prob = predictions[chId * outHeight * outWidth + rowId * outWidth + colId];
                        if (prob > maxProb) {
                            classId = chId;
                            maxProb = prob;
                        }
                    }
                }
                while (classId >= colors.size()) {
                    cv::Vec3b color(distr(rng), distr(rng), distr(rng));
                    colors.push_back(color);
                }
//                if(classId == 3)
//                    maskImg.at<cv::Vec3b>(rowId, colId) = cv::Vec3b(0, 0, 255);
                maskImg.at<cv::Vec3b>(rowId, colId) = colors[classId];
            }
        }
        cv::addWeighted(resImg, 0.7, maskImg, 0.3, 0, resImg);
        maskImg.copyTo(resImg);
//        cv::hconcat(resImg, maskImg, resImg);
        cv::resize(resImg, resImg, cv::Size(), 0.5, 0.5);
    }
    curr_infer_request.swap(next_infer_request);

}