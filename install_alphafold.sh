#!/bin/bash

# create, update and activate conda environment
source ~/.bashrc
conda create -n af2 python=3.8 -y
conda update -n base conda -y
conda init bash
conda activate af2

# install conda packages
conda install -y -c conda-forge openmm==7.5.1 cudnn==8.2.1.32 cudatoolkit==11.0.3 pdbfixer==1.7
conda install -y -c bioconda hmmer==3.3.2 hhsuite==3.3.0 kalign2==2.04

# install pip packages
pip install -r requirements.txt
pip install --upgrade "jax[cuda111]" -f \
    https://storage.googleapis.com/jax-releases/jax_releases.html \
    && pip3 install jaxlib==0.1.70+cuda111 -f \
    https://storage.googleapis.com/jax-releases/jax_releases.html

# update openmm
alphafold_path=$(pwd)
a=$(which python)
cd $(dirname $(dirname $a))/lib/python3.8/site-packages
patch -p0 < ${alphafold_path}/docker/openmm.patch

# download stereo_chemical_props.txt file
cd $alphafold_path
wget -q -P alphafold/common/ https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt
