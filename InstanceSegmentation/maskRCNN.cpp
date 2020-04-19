//
// Created by kun on 2020/4/18.
//

#include "maskRCNN.h"
#include "slog.hpp"
#include "ocv_common.hpp"
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>
#ifdef WITH_EXTENSIONS
#include <ext_list.hpp>
#endif

maskRCNN::maskRCNN():isFirstFrame(true) {

}

maskRCNN::~maskRCNN() {

}

bool maskRCNN::initiate(InferenceEngine::Core &ie, const std::string &model_path, const std::string &IE_device) {

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
              std::back_inserter(labels));
    this->network = this->networkReader.getNetwork();
    this->network.setBatchSize(1);
    slog::info << "Batch size is " << std::to_string(networkReader.getNetwork().getBatchSize()) << slog::endl;

    slog::info << "Preparing input blobs" << slog::endl;
    this->inputInfo = network.getInputsInfo();

    for (const auto & inputInfoItem : this->inputInfo) {
        if (inputInfoItem.second->getTensorDesc().getDims().size() == 4) {  // first input contains images
            this->InputName = inputInfoItem.first;
            inputInfoItem.second->setPrecision(Precision::U8);
        } else if (inputInfoItem.second->getTensorDesc().getDims().size() == 2) {  // second input contains image info
            inputInfoItem.second->setPrecision(Precision::FP32);
        } else {
            throw std::logic_error("Unsupported input shape with size = " + std::to_string(inputInfoItem.second->getTensorDesc().getDims().size()));
        }
    }

    const TensorDesc& inputDesc = inputInfo[this->InputName]->getTensorDesc();
    IE_ASSERT(inputDesc.getDims().size() == 4);

    this->input_height = getTensorHeight(inputDesc);
    this->input_width = getTensorWidth(inputDesc);
    this->InputSize = cv::Size(this->input_width, this->input_height);
    slog::info << "model input size is  " <<
               std::to_string(this->input_height)+" x "+std::to_string(this->input_width) << slog::endl;

    slog::info << "Preparing output blobs" << slog::endl;
//    this->network.addOutput("DetectionOutput", 0);
//    this->network.addOutput("DetectionOutput", 1);

    for(auto &item : this->outputInfo){
        item.second->setPrecision(Precision::FP32);
    }

    slog::info << "Loading model to the device" << slog::endl;
    this->executable_network = ie.LoadNetwork(network, IE_device);
    slog::info << "Create infer request" << slog::endl;
    this->curr_infer_request = this->executable_network.CreateInferRequestPtr();
    this->next_infer_request = this->executable_network.CreateInferRequestPtr();
}


bool maskRCNN::start_async_exec(cv::Mat &curr_frame, cv::Mat &next_frame, cv::Mat &result) {
    if(isFirstFrame){
        this->frameToBlob(curr_frame, curr_infer_request, this->inputInfo);
        curr_infer_request->StartAsync();
        isFirstFrame = false;
    } else{
        this->frameToBlob(next_frame, next_infer_request, this->inputInfo);
        next_infer_request->StartAsync();
    }

    if (OK == curr_infer_request->Wait(IInferRequest::WaitMode::RESULT_READY)){
        const auto masks_blob = curr_infer_request->GetBlob("raw_masks"); //6849/Unsqueeze
        const auto masks_data = masks_blob->buffer().as<float*>();

        const auto box_blob = curr_infer_request->GetBlob("boxes");    //DetectionOutput.0
        const auto box_data = box_blob->buffer().as<float*>();
        const auto cls_blob = curr_infer_request->GetBlob("classes");  //DetectionOutput.1
        const auto cls_data = cls_blob->buffer().as<float*>();
        const auto prob_blob = curr_infer_request->GetBlob("scores");  //DetectionOutput.2
        const auto prob_data = prob_blob->buffer().as<float*>();

        const float PROBABILITY_THRESHOLD = 0.2f;
        const float MASK_THRESHOLD = 0.5f;  // threshold used to determine whether mask pixel corresponds to object or to background

        IE_ASSERT(box_blob->getTensorDesc().getDims().size() == 2);
        size_t BOX_DESCRIPTION_SIZE = box_blob->getTensorDesc().getDims().back();
        auto aa =  box_blob->getTensorDesc().getDims();
        const TensorDesc& masksDesc = masks_blob->getTensorDesc();
        IE_ASSERT(masksDesc.getDims().size() == 4);
        size_t BOXES = getTensorBatch(masksDesc);
        size_t C = getTensorChannels(masksDesc);
        size_t H = getTensorHeight(masksDesc);
        size_t W = getTensorWidth(masksDesc);
        
        size_t box_stride = W * H * C;
        std::map<size_t, size_t> class_color;

        /** Iterating over all boxes **/
        for (size_t box = 0; box < BOXES; ++box) {

            float* box_info = box_data + box * BOX_DESCRIPTION_SIZE;
            float prob = prob_data[box];
            int class_id = static_cast<int>(cls_data[box]);

            float x1 = std::min(std::max(0.0f, box_info[0]), static_cast<float>(input_width));
            float y1 = std::min(std::max(0.0f, box_info[1]), static_cast<float>(input_height));
            float x2 = std::min(std::max(0.0f, box_info[2]), static_cast<float>(input_width));
            float y2 = std::min(std::max(0.0f, box_info[3]), static_cast<float>(input_height));

            int box_width = std::min(static_cast<int>(std::max(0.0f, x2 - x1)), static_cast<int>(input_width));
            int box_height = std::min(static_cast<int>(std::max(0.0f, y2 - y1)), static_cast<int>(input_height));

            if (prob > PROBABILITY_THRESHOLD) {
                size_t color_index = class_color.emplace(class_id, class_color.size()).first->second;
                auto& color = CITYSCAPES_COLORS[color_index % arraySize(CITYSCAPES_COLORS)];
                float* mask_arr = masks_data + box_stride * box + H * W * (class_id - 1);
                slog::info << "Detected class " << class_id << " with probability " << prob << " from batch " << 0
                           << ": [" << x1 << ", " << y1 << "], [" << x2 << ", " << y2 << "]" << slog::endl;
                cv::Mat mask_mat(H, W, CV_32FC1, mask_arr);

                cv::Rect roi = cv::Rect(static_cast<int>(x1), static_cast<int>(y1), box_width, box_height);
                cv::Mat roi_input_img = result(roi);
                const float alpha = 0.7f;

                cv::Mat resized_mask_mat(box_height, box_width, CV_32FC1);
                cv::resize(mask_mat, resized_mask_mat, cv::Size(box_width, box_height));

                cv::Mat uchar_resized_mask(box_height, box_width, CV_8UC3,
                                           cv::Scalar(color.blue(), color.green(), color.red()));
                roi_input_img.copyTo(uchar_resized_mask, resized_mask_mat <= MASK_THRESHOLD);

                cv::addWeighted(uchar_resized_mask, alpha, roi_input_img, 1.0f - alpha, 0.0f, roi_input_img);
                cv::rectangle(result, roi, cv::Scalar(0, 0, 1), 1);
            }
        }
//        std::string imgName = "out.png";
//        cv::imwrite(imgName, result);
//        slog::info << "Image " << imgName << " created!" << slog::endl;
        
    }
    curr_infer_request.swap(next_infer_request);

}

void maskRCNN::frameToBlob(cv::Mat &image, InferenceEngine::InferRequest::Ptr &infer_request, InputsDataMap &inputInfo) {
    for (const auto & inputInfoItem : inputInfo) {
        Blob::Ptr input = infer_request->GetBlob(inputInfoItem.first);

        /** Fill first input tensor with images. First b channel, then g and r channels **/
        if (inputInfoItem.second->getTensorDesc().getDims().size() == 4) {
            /** Iterate over all input images **/
            matU8ToBlob<unsigned char>(image, input, 0);
        }

        /** Fill second input tensor with image info **/
        if (inputInfoItem.second->getTensorDesc().getDims().size() == 2) {
            auto data = input->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
            data[0] = static_cast<float>(this->input_height);  // height
            data[1] = static_cast<float>(this->input_width);  // width
            data[2] = 1;
        }
    }
}