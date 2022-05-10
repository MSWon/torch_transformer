#!/bin/bash

MAIN_PATH="${YOUR_GIT_REPO_PATH}/torch_transformer"
MODEL_FILE_PATH="${MAIN_PATH}/koen_aihub_150k/model_150000.pt"
PACKAGE_NAME="koen.2022.0505"
DEVICE="cpu"
#DEVICE="cuda"

docker run -it \
    -v ${MAIN_PATH}:${MAIN_PATH} \
    -w ${MAIN_PATH} \
    pytorch:0.0.1 \
        python ${MAIN_PATH}/nmt/tools/export_graph.py \
                --model_path ${MODEL_FILE_PATH} \
                --package_name ${PACKAGE_NAME} \
                --device ${DEVICE}
