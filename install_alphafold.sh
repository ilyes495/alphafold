#!/bin/bash

conda create -n af2 python=3.8 -y
conda update -n base conda

source ~/anaconda3/etc/profile.d/conda.sh

conda activate af2

conda install -y -c conda-forge openmm==7.5.1 cudnn==8.2.1.32 cudatoolkit==11.0.3 pdbfixer==1.7
conda install -y -c bioconda hmmer==3.3.2 hhsuite==3.3.0 kalign2==2.04


pip install absl-py==0.13.0 biopython==1.79 chex==0.0.7 dm-haiku==0.0.4 dm-tree==0.1.6 immutabledict==2.0.0 jax==0.2.14 ml-collections==0.1.0 numpy==1.19.5 scipy==1.7.0 tensorflow==2.5.0

pip install --upgrade "jax[cuda111]" -f \
    https://storage.googleapis.com/jax-releases/jax_releases.html \
    && pip3 install jaxlib==0.1.70+cuda111 -f \
    https://storage.googleapis.com/jax-releases/jax_releases.html




# work_path=/path/to/alphafold-code
# work_path=.

# update openmm 
a=$(which python)
cd $(dirname $(dirname $a))/lib/python3.8/site-packages
patch -p0 < ./docker/openmm.patch

# Download stereo_chemical_props.txt file
wget -q -P ./alphafold/common/ https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt