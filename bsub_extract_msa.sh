#!/bin/bash

#BSUB -J jupyter
#BSUB -q gpuqueue
#BSUB -n 8
#BSUB -R rusage[mem=16]
#BSUB -W 10:00
#BSUB -gpu num=1
#BSUB -o %J.stdout
#BSUB -eo %J.stderr


cd $LS_SUBCWD
echo $LS_SUBCWD
source ~/.bashrc
conda --version
conda activate af2
export XDG_RUNTIME_DIR=""

python extract_MSA.py --input ../input_sequences/T080449_1.95d_SFPQ_Homo_sapiens.fasta