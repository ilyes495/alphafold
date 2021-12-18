import argparse
import itertools
import numpy as np
import os
import pickle

# parse input arguments
parser = argparse.ArgumentParser(description='Unpack representations saved by AF2.')
parser.add_argument('--input_folder', type=str, default=None,
                    help='Path to the input folder.')
parser.add_argument('--output_folder', type=str, default=None,
                    help='Path to the output folder.')
                    
# input and output folders
args = parser.parse_args()
input_folder = args.input_folder
output_folder = args.output_folder

# list of modules and iterations
module_list = ['msa', 'evoformer', 'structure']
num_iter = 4
module_iter_list = ['_iter_'.join(p) for p in itertools.product(module_list, np.arange(num_iter).astype(str))]

# create folders for modules and iterations output
for module_iter in module_iter_list:
  module_iter_path = os.path.join(output_folder, module_iter)
  if not os.path.exists(module_iter_path):
    os.makedirs(module_iter_path)

# iterate over AF2 output
input_rbr_folder_list = os.listdir(input_folder)
for input_rbr_folder in input_rbr_folder_list:
  rbr_name = input_rbr_folder.replace('__', '_')

  # pickle file paths
  processed_features_path = os.path.join(input_folder, input_rbr_folder, 'processed_features_model_1.pkl')
  result_path = os.path.join(input_folder, input_rbr_folder, 'result_model_1.pkl')
  
  # load pickle files
  with open(processed_features_path, 'rb') as f:
    processed_features = pickle.load(f)
  with open(result_path, 'rb') as f:
    result = pickle.load(f)
  
  # save MSA representations
  for i in range(num_iter):
    output_path = os.path.join(output_folder, 'msa_iter_{}'.format(i), rbr_name)
    np.save(output_path, processed_features['msa_feat'][i][0])
  
  # save evoformer representations
  for i in range(num_iter):
    output_path = os.path.join(output_folder, 'evoformer_iter_{}'.format(i), rbr_name)
    np.save(output_path, result['representations']['msa_first_row_iter_{}'.format(i)])
  
  # save structure module representations
  for i in range(num_iter):
    output_path = os.path.join(output_folder, 'structure_iter_{}'.format(i), rbr_name)
    np.save(output_path, result['representations']['structure_msa_first_row_iter_{}'.format(i)])
