# AlphaFold with saved intermediate representations

This repository contains modifications to [AlphaFold v2.0] (https://github.com/deepmind/alphafold), which saves 12 intermediate representations of the proteins predicted.

## Representations

The 12 representations are saved within `output/processed_features_model_1.pkl` and `output/result_model_1.pkl` by running the feature extraction scripts.

**Four output representations from featurization**
- Saved to `processed_features_model_1.pkl['msa_feat']`
- Dimensions: (4, s, r, 49). The first dimension indicates the 4 output representations. To obtain the representation corresponding to the predicted protein for each of the 4 representations, please extract the first row of dimension s (number of clusters). The resulting dimension per representation should be (r, 49).
  
**Four output representations from the Evoformer**
- Saved as `result_model_1.pkl['representations']['msa_first_row_iter_0-3']`
- Dimensions: (r, 256)
  
**Four output representations from the Structure Module** 
- Saved as `result_model_1.pkl['representations']['structure_msa_first_row_iter_0-3']`
- Dimensions: (r, 384)

The above extracted features could be unpacked by running the feature unpacking scripts.

## Installation

Run `install_alphafold.sh` to install this package. Conda is required.

## Running the script

The representation extraction process consists of two parts. First, to save the intermediate features from AF2. Then, to unpack those representations. A script to extract the MSA alone is also available.

### Feature extraction
`scripts/run_extract_features.py`: This script runs AlphaFold v2.0 and saves intermediate representations.

`bsub/bsub_extract_features.sh`: This script submits a jobarray for the above script on an HPC with an LSF scheduler.
    - `#BSUB -J run-alphafold[1-355]`: This indicates a job array for the 355 RBRs in the dataset.
    - `#BSUB -R rusage[mem=10]`: Please increase the memory requirement as needed.
    - `#BSUB -n 8`: Exactly 8 cores are required to run an AlphaFold job.
    - Please set `DATA_DIR` to the directory containing the AlphaFold database.
    - Please create a `logs/` folder. stdout and stderr files will be saved here.

### Feature unpacking
`scripts/run_unpack_features.py`: This script unpacks the output from `run_extract_features.py` and saves the 12 intermediate representations.

`bsub/bsub_unpack_features.sh`: This script submits a job for the above script on an HPC with an LSF scheduler.

### MSA extraction
`scripts/run_extract_msa.py`: This script runs AlphaFold v2.0 and saves the MSA.

`bsub/bsub/extract_msa.py`: This script submits a jobarray for the above script on an HPC with an LSF scheduler.
