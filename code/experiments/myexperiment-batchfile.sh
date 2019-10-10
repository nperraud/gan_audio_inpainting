#!/bin/bash -l
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=gpu
#SBATCH --output=myexperiment-o-%j.log
#SBATCH --error=myexperiment-e-%j-e.log
#SBATCH --account=sd01

module load daint-gpu
module load cray-python
module load TensorFlow/1.7.0-CrayGNU-18.08-cuda-9.1-python3

source $HOME/default/bin/activate

cd $SCRATCH/gan_audio_inpainting/code/experiments
srun python myexperiments.py
