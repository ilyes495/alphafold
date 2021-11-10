import sys
sys.path.append('/data/morrisq/baalii/alphafold/')

import os
os.environ['TF_FORCE_UNIFIED_MEMORY'] = '1'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '2.0'

"""Full AlphaFold protein structure prediction script."""
import json
import os
import pathlib
import pickle
import random
import time
from typing import Dict

from absl import app
from absl import flags
from absl import logging
from alphafold.common import protein
from alphafold.common import residue_constants
from alphafold.data import pipeline
from alphafold.data import templates
from alphafold.model import data
from alphafold.model import config
from alphafold.model import model
from alphafold.relax import relax
import numpy as np

import subprocess

import argparse

import jax
# if jax.local_devices()[0].platform == 'tpu':
#     raise RuntimeError('Colab TPU runtime not supported. Change it to GPU via Runtime -> Change Runtime Type -> Hardware accelerator -> GPU.')
# elif jax.local_devices()[0].platform == 'cpu':
#     raise RuntimeError('Colab CPU runtime not supported. Change it to GPU via Runtime -> Change Runtime Type -> Hardware accelerator -> GPU.')
    

parser = argparse.ArgumentParser(description='Extart MSA features for a given protein')
parser.add_argument('--input', type=str, help='path to the input sequence file', default='../data/input.fasta')

args = parser.parse_args()

MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 20

fasta_paths=[args.input]
output_dir='../alphafold_output'

data_dir= '../data' 
uniref90_database_path='../data/uniref90/uniref90.fasta'
mgnify_database_path='../data/mgnify/mgy_clusters_2018_12.fa'
small_bfd_database_path='../data/small_fbd/bfd-first_non_consensus_sequences.fasta'
pdb70_database_path='../data/pdb70/pdb70'
template_mmcif_dir='../data/pdb_mmcif/mmcif_files'
max_template_date='2020-05-14'
obsolete_pdbs_path='../data/pdb_mmcif/obsolete.dat'
preset='full_dbs'

bfd_database_path= '../data/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt'
uniclust30_database_path = '../data/uniclust30/uniclust30_2018_08/uniclust30_2018_08'

# Binary path (change me if required)
hhblits_binary_path='/home/baalii/anaconda3/envs/af2/bin/hhblits'
hhsearch_binary_path='/home/baalii/anaconda3/envs/af2/bin/hhsearch'
jackhmmer_binary_path='/home/baalii/anaconda3/envs/af2/bin/jackhmmer' 
kalign_binary_path='/home/baalii/anaconda3/envs/af2/bin/kalign' 


benchmark = False


use_small_bfd = preset == 'reduced_dbs'
if preset in ('reduced_dbs', 'full_dbs'):
    num_ensemble = 1
elif preset == 'casp14':
    num_ensemble = 8

    

# Check for duplicate FASTA file names.
fasta_names = [pathlib.Path(p).stem for p in fasta_paths]
print(fasta_names)
if len(fasta_names) != len(set(fasta_names)):
    raise ValueError('All FASTA paths must have a unique basename.')

print('All the checks passed!')    
template_featurizer = templates.TemplateHitFeaturizer(
  mmcif_dir=template_mmcif_dir,
  max_template_date=max_template_date,
  max_hits=MAX_TEMPLATE_HITS,
  kalign_binary_path=kalign_binary_path,
  release_dates_path=None,
  obsolete_pdbs_path=obsolete_pdbs_path)

data_pipeline = pipeline.DataPipeline(
  jackhmmer_binary_path=jackhmmer_binary_path,
  hhblits_binary_path=hhblits_binary_path,
  hhsearch_binary_path=hhsearch_binary_path,
  uniref90_database_path=uniref90_database_path,
  mgnify_database_path=mgnify_database_path,
  bfd_database_path=bfd_database_path,
  uniclust30_database_path=uniclust30_database_path,
  small_bfd_database_path=small_bfd_database_path,
  pdb70_database_path=pdb70_database_path,
  template_featurizer=template_featurizer,
  use_small_bfd=use_small_bfd)


output_dir = os.path.join(output_dir, fasta_names[0])
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
msa_output_dir = os.path.join(output_dir, 'msas')
if not os.path.exists(msa_output_dir):
    os.makedirs(msa_output_dir)

print('Output folders created!\nProcessing Starts...')
# Get features.
t_0 = time.time()
feature_dict = data_pipeline.process(
    input_fasta_path=fasta_paths[0],
    msa_output_dir=msa_output_dir)
timings['features'] = time.time() - t_0

print('Ruuning time: ', timings['features'])

print('saving the results!')

# Write out features as a pickled dictionary.
features_output_path = os.path.join(output_dir, 'features.pkl')
with open(features_output_path, 'wb') as f:
    pickle.dump(feature_dict, f, protocol=4)