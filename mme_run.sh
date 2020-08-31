#!/bin/bash

# Configure the resources required
#SBATCH -p batch                                                # partition (this is the queue your job will be added to)
#SBATCH -n 1              	                                # number of tasks (sequential job starts 1 task) (check this if your job unexpectedly uses 2 nodes)
#SBATCH -c 2              	                                # number of cores (sequential job calls a multi-thread program that uses 8 cores)
#SBATCH --time=24:00:00                                         # time allocation, which has the format (D-HH:MM), here set to 1 hour
#SBATCH --gres=gpu:1                                            # generic resource required (here requires 4 GPUs)
#SBATCH --mem=33GB                                              # specify memory required per node (here set to 16 GB)

# Configure notifications 
#SBATCH --mail-type=END                                         # Send a notification email when the job is done (=END)
#SBATCH --mail-type=FAIL                                        # Send a notification email when the job fails (=FAIL)
##SBATCH --mail-user=a1227514@adelaide.edu.au          # Email to which notifications will be sent
#SBATCH -M volta

module load Anaconda3/5.0.1
module load CUDA/9.0.176
module load cuDNN/7.3.1-CUDA-9.0.176


source /home/a1227514/wangxu/anaconda3/bin/activate py3.6_pt1.0_cuda9

# python ./train_config.py --config_file ./configs/1_3.yml
METHOD='MME'
NET='resnet34'
python3 main.py --method ${METHOD} --dataset multi --source real --target sketch --net ${NET} --save_check
#python test.py
source /home/a1227514/wangxu/anaconda3/bin/deactivate



