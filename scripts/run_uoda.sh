#!/bin/bash
METHOD='UODA'
NET='resnet34'
python3 main_UODA2.py --method ${METHOD} --dataset multi --source real --target sketch --net ${NET} --save_check

