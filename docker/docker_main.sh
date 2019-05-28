#!/usr/bin/env bash

docker run -u $(id -u):$(id -g)
           -v /home/cs4li/Dev/deep_ekf_vio:/scratch/cs4li/deep_ekf_vio \
           -v /home/cs4li/Dev/KITTI:/scratch/cs4li/KITTI \
           -v /home/cs4li/Dev/EUROC:/scratch/cs4li/EUROC \
           --shm-size 128g --runtime=nvidia --rm cs4li_deep_ekf_vio \
           python3 /home/cs4li/Dev/deep_ekf_vio/main.py
           --description "test full image gloss no ekf" --gpu_id 0 1