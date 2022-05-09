#!/bin/bash

#####################
# Parameter Setting #
#####################

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64

## TRAIN ##
# corpus_config and model_config for training
NUM_GPU=8

MAIN_PATH="/home1/irteam/users/minsub/for_dev/minsub_git/torch_transformer"

CORPUS_PATH_ON_HOST="${MAIN_PATH}/train_data"
CORPUS_PATH_ON_DOCKER="${MAIN_PATH}/train_data"

CONFIG_PATH_ON_HOST="${MAIN_PATH}/config"
CONFIG_PATH_ON_DOCKER="${MAIN_PATH}/config"
CONFIG_FILE_PATH="${CONFIG_PATH_ON_DOCKER}/train_config.yaml"
SAVED_PATH="${MAIN_PATH}/enko_aihub_150k"

##############
# Run Script #
##############

# learn with docker directly
docker run -it \
    -v ${MAIN_PATH}:${MAIN_PATH} \
    -v ${CORPUS_PATH_ON_HOST}:${CORPUS_PATH_ON_DOCKER} \
    -v ${CONFIG_PATH_ON_HOST}:${CONFIG_PATH_ON_DOCKER} \
    -w ${MAIN_PATH} \
    pytorch:0.0.1 \
        mpiexec --allow-run-as-root -x NCCL_DEBUG=INFO -np ${NUM_GPU} \
            python ${MAIN_PATH}/nmt/train/main.py \
                    --config ${CONFIG_FILE_PATH} \
                    --saved_path ${SAVED_PATH}
