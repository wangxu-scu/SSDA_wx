#!/bin/sh
CUDA_VISIBLE_DEVICES=$1 python3 main_UODA2.py --method $2 --dataset multi --source real --target sketch --net $3 --save_check
