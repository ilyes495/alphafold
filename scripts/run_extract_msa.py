"""Full AlphaFold protein MSA extraction script."""
import argparse
import os
import pathlib
import pickle
import random
import shutil
import sys
import time

sys.path.append('../')
from absl import logging
from alphafold.data import pipeline
from alphafold.data import templates
from alphafold.data.tools import hhsearch

# Configure Logging.
logging.set_verbosity(logging.INFO)

# Parse input arguments.
parser = argparse.ArgumentParser(description='Extract MSA for a given protein')
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
use_precomputed_msas = False

# Set global variables.
MAX_TEMPLATE_HITS = 20

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

logging.info('Final timings for %s: %s', fasta_name, timings)
