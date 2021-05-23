#!/bin/sh

die () {
    echo >&2 "$@"
    exit 1
}

[ "$#" -eq 1 ] || die "1 argument required (MegaDepth path), $# provided"

export MEGADEPTH=$1
export DEBUG=0

# python train.py --gpu_idx 0 --dataset megadepth --model-type stylize_rgb_2steps \
#     --feature_model ./pretrainedmodels/effnetB1_ep86.pth --batch-size 5 --use_discriminator 1 --use_pretrained 1 \
#     --lambda_feat 0.01 --W 512 --losses hinge --resume 1 --WCoarse 128 --hinge_samples 3 --lambda2 0.001 --num_workers 5 \
#     --dataset megadepth --lr 0.0001 --dataset_base_path $1 --network unet_256 --use_mask 1 --topk 1 --use_weights 1

python train.py --model-type fpn_conf_broad --losses hinge --WCoarse 128 --use_pretrained 1 --jitter 0 --keep_aspect 0  --dataset megadepth --network resnet50\
                --batch-size 8 --lr 0.0001 --train_end2end 0 --hard_samples 512 --hinge_samples 3\
                --W 256 --num_workers 2 --dataset_base_path $MEGADEPTH