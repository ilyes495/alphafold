#!/bin/bash
#BSUB -J unpack_features
#BSUB -W 8:00
#BSUB -R rusage[mem=10]
#BSUB -n 8
#BSUB -o ../logs/unpack_features_%J.stdout
#BSUB -eo ../logs/unpack_features_%J.stderr

# activate conda environment
source ~/.bashrc
conda activate af2

# set directories
INPUT_DIR=/data/morrisq/alphafold/rbr_seqs_all/af2_output/
OUTPUT_DIR=/data/morrisq/alphafold/rbr_seqs_all/af2_representation/


# run the command
python3 ../scripts/run_unpack_features.py \
--input_folder=$INPUT_DIR \
--output_folder=$OUTPUT_DIR
