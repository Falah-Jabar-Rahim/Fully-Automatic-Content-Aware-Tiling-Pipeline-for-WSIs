#!/bin/bash
name="norm_mixed"
checkpoints_dir_path=""
emnist_img_path=""
samedomain_img_path=""
tiger_dataset=""
slide_num_path=""

nohup python train.py \
 --name $name \
 --direction AtoB \
 --use_wandb \
 --dataset_mode mixed \
 --gpu_ids 0 \
 --checkpoints_dir $checkpoints_dir_path \
 --model pix2pix \
 --batch_size 2 \
 --do_norm \
 --load_size 256 \
 --image_pth $samedomain_img_path \
 --template_pth $emnist_img_path \
 --tiger_image_pth $tiger_dataset \
 --preprocess none > ./Results/log_$name.out &