#!/bin/bash

#BSUB -J jupyter
#BSUB -q gpuqueue
#BSUB -m ly-gpu
#BSUB -n 8
#BSUB -R rusage[mem=32] 
#BSUB -W 72:00
#BSUB -gpu num=2
#BSUB -o %J.stdout
#BSUB -eo %J.stderr


cd $LS_SUBCWD
echo $LS_SUBCWD
source ~/.bashrc
conda --version
conda activate af2
export XDG_RUNTIME_DIR=""
port=8889
node=$(hostname -s)
user=$(whoami)
cluster=$(hostname)

# print tunneling instructions jupyter-log
echo -e " MacOS or linux terminal command to create your ssh tunnel ssh -N -L ${port}:${node}:${port} ${user}@${cluster}"
jupyter lab --no-browser --port=${port} --ip=${node}
