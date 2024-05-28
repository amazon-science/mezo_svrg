#!/bin/sh
#!/usr/bin/env bash
#
set -ex

export CONTAINER_NAME=${1:-"zhuha_nightly"}
export CONTAINER_IMAGE=${2:-"pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel"}
# export CONTAINER_IMAGE=${2:-"pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel"}

# Set the docker ECR path to your DLC
# aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 855988369404.dkr.ecr.us-west-2.amazonaws.com

FSX_MOUNT="-v /fsx/:/fsx"
NFS_MOUNT="-v /nfs/:/nfs"

docker stop $(docker ps -q -a) || true
docker rm $(docker ps -q -a) || true

docker pull ${CONTAINER_IMAGE}

# TODO : Add the GDR copy device
docker run --name $CONTAINER_NAME --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm \
    --net=host \
    -e NCCL_SOCKET_IFNAME=ens5 \
    -e NCCL_SOCKET_IFNAME="^lo,docker" \
    -e RDMAV_FORK_SAFE=1 \
    -e GPU_NUM_DEVICES=8 \
    -e FI_EFA_USE_DEVICE_RDMA=1  \
    -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
    -e PYTHONPATH=/workspace/gpu_llama \
    --security-opt seccomp=unconfined \
    --privileged \
    --shm-size=561g \
    -d \
    ${FSX_MOUNT} \
    ${NFS_MOUNT} \
    ${CONTAINER_IMAGE} 

# Allow containers to talk to each other
docker exec -itd ${CONTAINER_NAME} bash -c "printf \"Port 2022\n\" >> /etc/ssh/sshd_config"
docker exec -itd ${CONTAINER_NAME} bash -c "printf \"  Port 2022\n\" >> /root/.ssh/config"
docker exec -itd ${CONTAINER_NAME} bash -c "service ssh start"

docker exec ${CONTAINER_NAME} pip3 install --force-reinstall --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

docker exec ${CONTAINER_NAME} pip3 install -r /fsx/users/zhuha/ZO_SmallScaleExp/requirements.txt

EXP_PATH=/fsx/users/zhuha/ZO_SmallScaleExp/results/robertalarge-$(date "+%y-%m-%d-%H-%M-%S")
mkdir -p $EXP_PATH
echo $EXP_PATH

# gpu=$1
# task=$2
# alg=$3
# an=$4
# lr=$5
# lr_mb=$6
# PARTIAL=$7
# EXP_PATH=$8
# LAST=$9

docker exec ${CONTAINER_NAME} /bin/bash /fsx/users/zhuha/ZO_SmallScaleExp/finetune_llm_full.sh 0 mnli ZO 4 1e-6 1e-5 full $EXP_PATH f y
docker exec ${CONTAINER_NAME} /bin/bash /fsx/users/zhuha/ZO_SmallScaleExp/finetune_llm_full.sh 1 mnli ZOSVRG 5 1e-4 1e-6 partial $EXP_PATH f y
docker exec ${CONTAINER_NAME} /bin/bash /fsx/users/zhuha/ZO_SmallScaleExp/finetune_llm_full.sh 2 mnli ZOSVRG 5 1e-5 1e-5 partial $EXP_PATH f y
docker exec ${CONTAINER_NAME} /bin/bash /fsx/users/zhuha/ZO_SmallScaleExp/finetune_llm_full.sh 3 mnli ZOSVRG 4 1e-5 1e-6 full $EXP_PATH f y
docker exec ${CONTAINER_NAME} /bin/bash /fsx/users/zhuha/ZO_SmallScaleExp/finetune_llm_full.sh 4 mnli ZO 5 1e-6 1e-5 partial $EXP_PATH f y
docker exec ${CONTAINER_NAME} /bin/bash /fsx/users/zhuha/ZO_SmallScaleExp/finetune_llm_full.sh 5 mnli ZO 5 1e-6 1e-6 full $EXP_PATH f y
docker exec ${CONTAINER_NAME} /bin/bash /fsx/users/zhuha/ZO_SmallScaleExp/finetune_llm_full.sh 6 mnli ZO 5 1e-6 1e-5 partial $EXP_PATH f y
docker exec ${CONTAINER_NAME} /bin/bash /fsx/users/zhuha/ZO_SmallScaleExp/finetune_llm_full.sh 7 mnli ZOSVRG 3 1e-5 1e-6 full $EXP_PATH t y

wait 
