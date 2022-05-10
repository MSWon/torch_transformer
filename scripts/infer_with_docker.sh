#!/bin/bash
MAIN_PATH="${YOUR_GIT_REPO_PATH}/torch_transformer"
PACKAGE_PATH="${MAIN_PATH}/koen.2022.0505"
SRC_LANG="ko"
TGT_LANG="en"
BATCH_SIZE="4"
CORPUS_PATH="${MAIN_PATH}/test_data/test.ko"
DEVICE="cpu"

docker run -it \
    -v ${MAIN_PATH}:${MAIN_PATH} \
    -w ${MAIN_PATH} \
    pytorch:0.0.1 \
        python ${MAIN_PATH}/nmt/inference/main.py --package_path ${PACKAGE_PATH} \
                                                  --src_lang ${SRC_LANG} \
                                                  --tgt_lang ${TGT_LANG} \
                                                  --batch_size ${BATCH_SIZE} \
                                                  --corpus_path ${CORPUS_PATH} \
                                                  --device ${DEVICE}
