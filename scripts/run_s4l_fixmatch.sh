#!/bin/bash
METHOD='S4L_FIXMATCH'
NET='resnet34'
python3 main_S4L_Fixmatch_SSDA.py --method ${METHOD} --dataset multi --source real --target sketch --net ${NET} --save_check

