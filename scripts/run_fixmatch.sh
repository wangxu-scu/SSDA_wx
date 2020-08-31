#!/bin/bash
METHOD='FIXMATCH'
NET='resnet34'
python3 main_FixMatchSSDA.py --method ${METHOD} --dataset multi --source real --target sketch --net ${NET} --save_check

