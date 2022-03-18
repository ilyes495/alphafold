"""Full AlphaFold protein feature extraction script."""
import argparse
import json
import os
import pathlib
import pickle
import random
import shutil
import sys
import time
from typing import Dict, Union, Optional

import jax
import numpy as np

sys.path.append('../')
from absl import app
from absl import flags
from absl import logging
from alphafold.common import protein
from alphafold.common import residue_constants
from alphafold.data import pipeline
from alphafold.data import pipeline_multimer
from alphafold.data import templates
from alphafold.data.tools import hhsearch
from alphafold.data.tools import hmmsearch
from alphafold.model import config
from alphafold.model import data
from alphafold.model import model
from alphafold.relax import relax

# Check for GPU availability.
if jax.local_devices()[0].platform == 'tpu':
    raise RuntimeError('TPU runtime not supported. Change it to GPU via Runtime -> Change Runtime Type -> Hardware accelerator -> GPU.')
elif jax.local_devices()[0].platform == 'cpu':
    raise RuntimeError('CPU runtime not supported. Change it to GPU via Runtime -> Change Runtime Type -> Hardware accelerator -> GPU.')

# Configure Logging.
logging.set_verbosity(logging.INFO)

# Parse input arguments.
parser = argparse.ArgumentParser(description='Extract features for a given protein')
parser.add_argument('--fasta_path', type=str, default=None,
                    help='Path to a FASTA file.')
parser.add_argument('--data_dir', type=str, default=None,
                    help='Path to directory of supporting data.')
parser.add_argument('--output_dir', type=str, default=None,
                    help='Path to a directory that will store the results.')
parser.add_argument('--jackhmmer_binary_path', type=str, default=shutil.which('jackhmmer'),
                    help='Path to the JackHMMER executable.')
parser.add_argument('--hhblits_binary_path', type=str, default=shutil.which('hhblits'),
                    help='Path to the HHblits executable.')
parser.add_argument('--hhsearch_binary_path', type=str, default=shutil.which('hhsearch'),
                    help='Path to the HHsearch executable.')
parser.add_argument('--kalign_binary_path', type=str, default=shutil.which('kalign'),
                    help='Path to the Kalign executable.')
parser.add_argument('--uniref90_database_path', type=str, default=None,
                    help='Path to the Uniref90 database for use by JackHMMER.')
parser.add_argument('--mgnify_database_path', type=str, default=None,
                    help='Path to the MGnify database for use by JackHMMER.')                    
parser.add_argument('--bfd_database_path', type=str, default=None,
                    help='Path to the BFD database for use by HHblits.') 
parser.add_argument('--small_bfd_database_path', type=str, default=None,
                    help='Path to the small version of BFD used with the "reduced_dbs" preset.')
parser.add_argument('--uniclust30_database_path', type=str, default=None,
                    help='Path to the uniclust30 database for use by HHblits.')                    
parser.add_argument('--pdb70_database_path', type=str, default=None,
                    help='Path to the PDB70 database for use by HHsearch.')  
parser.add_argument('--template_mmcif_dir', type=str, default=None,
                    help='Path to a directory with template mmCIF structures, '
                    'each named <pdb_id>.cif')
parser.add_argument('--max_template_date', type=str, default=None,
                    help='Maximum template release date to consider. Important if '
                    'folding historical test sets.')
parser.add_argument('--obsolete_pdbs_path', type=str, default=None,
                    help='Path to file containing a mapping from obsolete PDB IDs to the '
                    'PDB IDs of their replacements.')
parser.add_argument('--db_preset', type=str, default='full_dbs',
                    help='Choose preset MSA database configuration - '
                    'smaller genetic database config (reduced_dbs) or '
                    'full genetic database config (full_dbs)')
parser.add_argument('--model_name', type=str, default='model_1',
                    help='Name of the model preset.')
args = parser.parse_args()
fasta_path = args.fasta_path
data_dir = args.data_dir
output_dir = args.output_dir
jackhmmer_binary_path = args.jackhmmer_binary_path
hhblits_binary_path = args.hhblits_binary_path
hhsearch_binary_path = args.hhsearch_binary_path
kalign_binary_path = args.kalign_binary_path
uniref90_database_path = args.uniref90_database_path
mgnify_database_path = args.mgnify_database_path
bfd_database_path = args.bfd_database_path
small_bfd_database_path = args.small_bfd_database_path
uniclust30_database_path = args.uniclust30_database_path
pdb70_database_path = args.pdb70_database_path
template_mmcif_dir = args.template_mmcif_dir
max_template_date = args.max_template_date
obsolete_pdbs_path = args.obsolete_pdbs_path
db_preset = args.db_preset
model_names = [args.model_name]
benchmark = False
random_seed = None
use_precomputed_msas = False

# Set global variables.
MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 20

# Determine the number of ensembles.
use_small_bfd = db_preset == 'reduced_dbs'
if db_preset in ('reduced_dbs', 'full_dbs'):
    num_ensemble = 1
elif db_preset == 'casp14':
    num_ensemble = 8

# Configure HHSearch.
template_searcher = hhsearch.HHSearch(
    binary_path=hhsearch_binary_path,
    databases=[pdb70_database_path])
template_featurizer = templates.HhsearchHitFeaturizer(
    mmcif_dir=template_mmcif_dir,
    max_template_date=max_template_date,
    max_hits=MAX_TEMPLATE_HITS,
    kalign_binary_path=kalign_binary_path,
    release_dates_path=None,
    obsolete_pdbs_path=obsolete_pdbs_path)

# Perform MSA.
data_pipeline = pipeline.DataPipeline(
    jackhmmer_binary_path=jackhmmer_binary_path,
    hhblits_binary_path=hhblits_binary_path,
    uniref90_database_path=uniref90_database_path,
    mgnify_database_path=mgnify_database_path,
    bfd_database_path=bfd_database_path,
    uniclust30_database_path=uniclust30_database_path,
    small_bfd_database_path=small_bfd_database_path,
    template_searcher=template_searcher,
    template_featurizer=template_featurizer,
    use_small_bfd=use_small_bfd,
    use_precomputed_msas=use_precomputed_msas)

# Configure model.
model_runners = {}
for model_name in model_names:
    model_config = config.model_config(model_name)
    model_config.data.eval.num_ensemble = num_ensemble
    model_params = data.get_model_haiku_params(
        model_name=model_name, data_dir=data_dir)
    model_runner = model.RunModel(model_config, model_params)
    model_runners[model_name] = model_runner
logging.info('Have %d models: %s', len(model_runners),
             list(model_runners.keys()))

# Configure amber relaxer.
amber_relaxer = relax.AmberRelaxation(
  max_iterations=RELAX_MAX_ITERATIONS,
  tolerance=RELAX_ENERGY_TOLERANCE,
  stiffness=RELAX_STIFFNESS,
  exclude_residues=RELAX_EXCLUDE_RESIDUES,
  max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS)

# Generate random seed.
if random_seed is None:
    random_seed = random.randrange(sys.maxsize)
logging.info('Using random seed %d for the data pipeline', random_seed)

# Make output folder.
timings = {}
fasta_name = pathlib.Path(fasta_path).stem
output_dir = os.path.join(output_dir, fasta_name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
msa_output_dir = os.path.join(output_dir, 'msas')
if not os.path.exists(msa_output_dir):
    os.makedirs(msa_output_dir)

# Get features.
t_0 = time.time()
features_output_path = os.path.join(output_dir, 'features.pkl')
if os.path.exists(features_output_path):
    with open(features_output_path, 'rb') as f:
        feature_dict = pickle.load(f)
else:
    feature_dict = data_pipeline.process(
        input_fasta_path=fasta_path,
        msa_output_dir=msa_output_dir)
    with open(features_output_path, 'wb') as f:
        pickle.dump(feature_dict, f, protocol=4)
timings['features'] = time.time() - t_0

# Run the models.
unrelaxed_pdbs = {}
relaxed_pdbs = {}
ranking_confidences = {}
num_models = len(model_runners)
for model_index, (model_name, model_runner) in enumerate(model_runners.items()):
    print(f'Running model {model_name}')
    t_0 = time.time()
    model_random_seed = model_index + random_seed * num_models
    processed_feature_dict = model_runner.process_features(
        feature_dict, random_seed=model_random_seed)
        
    # Save the processed features.
    processed_features_output_path = os.path.join(output_dir, f'processed_features_{model_name}.pkl')
    with open(processed_features_output_path, 'wb') as f:
        pickle.dump(processed_feature_dict, f, protocol=4)
    timings[f'process_features_{model_name}'] = time.time() - t_0
    t_0 = time.time()
    
    # Run the prediction.
    prediction_result = model_runner.predict(processed_feature_dict,
                                             random_seed=model_random_seed)
    t_diff = time.time() - t_0
    timings[f'predict_and_compile_{model_name}'] = t_diff
    logging.info(
        'Total JAX model %s on %s predict time (includes compilation time, see --benchmark): %.1fs',
        model_name, fasta_name, t_diff)    
    plddt = prediction_result['plddt']
    ranking_confidences[model_name] = prediction_result['ranking_confidence']

    # Save the model outputs.
    result_output_path = os.path.join(output_dir, f'result_{model_name}.pkl')
    with open(result_output_path, 'wb') as f:
        pickle.dump(prediction_result, f, protocol=4)
    
    # Add the predicted LDDT in the b-factor column.
    # Note that higher predicted LDDT value means higher model confidence.
    plddt_b_factors = np.repeat(
        plddt[:, None], residue_constants.atom_type_num, axis=-1)
    unrelaxed_protein = protein.from_prediction(
        features=processed_feature_dict,
        result=prediction_result,
        b_factors=plddt_b_factors,
        remove_leading_feature_dimension=not model_runner.multimer_mode)

    # Save unrelaxed structure
    unrelaxed_pdbs[model_name] = protein.to_pdb(unrelaxed_protein)
    unrelaxed_pdb_path = os.path.join(output_dir, f'unrelaxed_{model_name}.pdb')
    with open(unrelaxed_pdb_path, 'w') as f:
        f.write(unrelaxed_pdbs[model_name])
   
    # Save relaxed structure
    if amber_relaxer:
        # Relax the prediction.
        t_0 = time.time()
        relaxed_pdb_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein)
        timings[f'relax_{model_name}'] = time.time() - t_0
        relaxed_pdbs[model_name] = relaxed_pdb_str

        # Save the relaxed PDB.
        relaxed_output_path = os.path.join(
            output_dir, f'relaxed_{model_name}.pdb')
        with open(relaxed_output_path, 'w') as f:
            f.write(relaxed_pdb_str)

# Rank by model confidence and write out relaxed PDBs in rank order.
ranked_order = []
for idx, (model_name, _) in enumerate(
    sorted(ranking_confidences.items(), key=lambda x: x[1], reverse=True)):
    ranked_order.append(model_name)
    ranked_output_path = os.path.join(output_dir, f'ranked_{idx}.pdb')
    with open(ranked_output_path, 'w') as f:
        if amber_relaxer:
            f.write(relaxed_pdbs[model_name])
        else:
            f.write(unrelaxed_pdbs[model_name])

ranking_output_path = os.path.join(output_dir, 'ranking_debug.json')
with open(ranking_output_path, 'w') as f:
    label = 'iptm+ptm' if 'iptm' in prediction_result else 'plddts'
    f.write(json.dumps(
        {label: ranking_confidences, 'order': ranked_order}, indent=4))

logging.info('Final timings for %s: %s', fasta_name, timings)

timings_output_path = os.path.join(output_dir, 'timings.json')
with open(timings_output_path, 'w') as f:
    f.write(json.dumps(timings, indent=4))
