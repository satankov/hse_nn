# 4-Potts uni project
HSE | Graduate | 4-Potts group

It is 4-state Potts group project.
My drafts and common scripts are placed here.

## Monte Carlo 
Each model directory contains script to simulate Monte Carlo and generate data (lattice snapshots) for future neural network training.

#### Quickstart:
1. Compile cython extension. 
> python3 setup.py build_ext --inplace
2. Set temperature range and tc `t_range(tc)` in `generate_data.py`, lattice size `L` and other parameters.
3. Correct settings in `run.sh` bash file.
4. Call bash file
> sbatch run.sh

## Neural Network
Neural network structure in .ipynb file.
