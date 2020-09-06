//
// Created by kun on 2020/5/30.
//

#include "classifications.h"
#include "common.hpp"
#include "slog.hpp"
#include "helper.h"
#include <inference_engine.hpp>
#ifdef WITH_EXTENSIONS
#include <ext_list.hpp>
#endif

using namespace InferenceEngine;

classifications::classifications():isFirstFrame(true) {

}

classifications::~classifications() {

}

bool classifications::initiate(InferenceEngine::Core &ie, const std::string &model_path,
                               const std::string &IE_device) {

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
              std::back_inserter(labels));
    this->network = this->networkReader.getNetwork();
    this->network.setBatchSize(1);
    slog::info << "Batch size is " << std::to_string(networkReader.getNetwork().getBatchSize()) << slog::endl;

    slog::info << "Preparing input blobs" << slog::endl;
    InputsDataMap inputInfo(network.getInputsInfo());
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
    this->OutputName = this->outputInfo.begin()->first;
    //auto out_layer = network.getLayerByName(this->OutputName.c_str());
    this->num_classes = this->network.getLayerByName(this->OutputName.c_str())->GetParamAsInt("num_classes");
//    num_classes = 2;
    if (static_cast<int>(labels.size()) != num_classes) {
        if (static_cast<int>(labels.size()) == (num_classes - 1)){
            labels.insert(labels.begin(), "fake");
        }else {
            labels.assign(num_classes, "");
        }
    }
    DataPtr outputData = outputInfo.begin()->second;
    outputData->setPrecision(Precision::FP32);

    slog::info << "Loading model to the device" << slog::endl;
    executable_network = ie.LoadNetwork(network, IE_device, {});
    slog::info << "Create infer request" << slog::endl;
    curr_infer_request = executable_network.CreateInferRequestPtr();
    next_infer_request = executable_network.CreateInferRequestPtr();

}


bool classifications::start_async_exec(cv::Mat &curr_frame, cv::Mat &next_frame, cv::Mat &result) {

    if(isFirstFrame){
        helper::frameToBlob(curr_frame, curr_infer_request, InputName);
        curr_infer_request->StartAsync();
        isFirstFrame = false;
    } else{
        helper::frameToBlob(next_frame, next_infer_request, InputName);
        next_infer_request->StartAsync();
    }

    if (OK == curr_infer_request->Wait(IInferRequest::WaitMode::RESULT_READY)){
        const Blob::Ptr output_blob = curr_infer_request->GetBlob(OutputName);
        const float* output = output_blob->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();

    }
    curr_infer_request.swap(next_infer_request);
    //next_frame.copyTo(curr_frame);
    return true;
}

bool classifications::start_sync_exec(cv::Mat &curr_frame, cv::Mat &result) {
    return false;
}
