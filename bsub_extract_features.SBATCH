#!/bin/bash
#SBATCH --job-name=alphafold-run
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=60GB
#SBATCH --gres=gpu
#SBATCH --error=slurm-%A_%a.out

# activate conda environment
source ~/.bashrc
conda activate /scratch/$USER/envs/af2

# set environmental variables
export TF_FORCE_UNIFIED_MEMORY=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=4.0

# set directories
DATA_DIR=/vast/work/public/alphafold
INPUT_DIR=input/
OUTPUT_DIR=output/

# get input file
INPUT_FILES=($INPUT_DIR/*)
INPUT_FILE=${INPUT_FILES[$(($SLURM_ARRAY_TASK_ID))]}
echo $INPUT_FILE

# run the command
python3 run_extract_features.py \
--fasta_path=$INPUT_FILE \
--data_dir=$DATA_DIR \
--output_dir=$OUTPUT_DIR \
--uniref90_database_path=$DATA_DIR/uniref90/uniref90.fasta \
--uniclust30_database_path=$DATA_DIR/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
--mgnify_database_path=$DATA_DIR/mgnify/mgy_clusters.fa \
--pdb70_database_path=$DATA_DIR/pdb70/pdb70 \
--template_mmcif_dir=$DATA_DIR/pdb_mmcif/mmcif_files \
--max_template_date=2020-05-14 \
--obsolete_pdbs_path=$DATA_DIR/pdb_mmcif/obsolete.dat \
--model_name='model_1' \
--bfd_database_path=$DATA_DIR/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
--small_bfd_database_path=$DATA_DIR/small_bfd/bfd-first_non_consensus_sequences.fasta \
--db_preset='full_dbs'
