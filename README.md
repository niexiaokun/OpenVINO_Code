# OpenVINO_Code

convert tensorflow object detection models:

openvino_dir=/opt/intel/openvino
mo_dir=${openvino_dir}/deployment_tools/model_optimizer
model_dir=`pwd`
data_type=FP16

python3 ${mo_dir}/mo_tf.py       \
        --input_model=${model_dir}/frozen_inference_graph.pb        \
        --output=detection_boxes,detection_scores,num_detections        \
        --data_type=${data_type}   \
        --tensorflow_use_custom_operations_config=${mo_dir}/extensions/front/tf/ssd_v2_support.json   \
        --tensorflow_object_detection_api_pipeline_config=${model_dir}/pipeline.config   \
        --input=image_tensor \
        --reverse_input_channels \
        --input_shape=[1,300,300,3]


python3 ${mo_dir}/mo_tf.py      \
        --input_model=${model_dir}/frozen_inference_graph.pb      \
        --data_type=${data_type}     \
        --tensorflow_use_custom_operations_config=${mo_dir}/extensions/front/tf/mask_rcnn_support.json  \
        --tensorflow_object_detection_api_pipeline_config=${model_dir}/pipeline.config  \
        --reverse_input_channels  \
        --input=image_tensor   \
        --input_shape=[1,800,1365,3]
