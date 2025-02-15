#!/bin/bash

name="norm_mixed"
version="" #The version denotes the directory where you want to load calculated results from, the directory are named as {versionid}_test_latest
results_dir_path=""

nohup python calc_metrics.py \
 --model pix2pix \
 --dataset_mode pairink \
 --version $version \
 --results_dir $results_dir_path \
 --name $name > ./Results/logmetric_$name.out &