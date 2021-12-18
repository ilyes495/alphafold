#!/bin/bash
#BSUB -J run_alphafold[1-497]
#BSUB -q gpuqueue
#BSUB -W 8:00
#BSUB -R rusage[mem=60]
#BSUB -gpu "num=1"
#BSUB -n 8
#BSUB -o logs/run_alphafold_%J.stdout
#BSUB -eo logs/run_alphafold%J.stderr

# activate conda environment
source ~/.bashrc
conda activate af2

# set environmental variables
export TF_FORCE_UNIFIED_MEMORY=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=4.0

# set directories
DATA_DIR=/data/alphafold-db/
INPUT_DIR=/data/morrisq/alphafold/rbr_seq_15/rbr_seq/
OUTPUT_DIR=/data/morrisq/alphafold/rbr_seq_15/af2_output/

# get input file
INPUT_FILES=($INPUT_DIR/*)
INPUT_FILE=${INPUT_FILES[$((LSB_JOBINDEX - 1))]}
echo $INPUT_FILE

# run the command
python3 run_extract_features.py \
--fasta_path=$INPUT_FILE \
--data_dir=$DATA_DIR \
--output_dir=$OUTPUT_DIR \
--uniref90_database_path=$DATA_DIR/uniref90/uniref90.fasta \
--mgnify_database_path=$DATA_DIR/mgnify/mgy_clusters_2018_12.fa \
--bfd_database_path=$DATA_DIR/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
--uniclust30_database_path=$DATA_DIR/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
--pdb70_database_path=$DATA_DIR/pdb70/pdb70 \
--template_mmcif_dir=$DATA_DIR/pdb_mmcif/mmcif_files \
--max_template_date=2020-05-14 \
--obsolete_pdbs_path=$DATA_DIR/pdb_mmcif/obsolete.dat \
--db_preset='full_dbs' \
--model_name='model_1'
