#!/bin/sh

set -e
#$ -N AccNet3          
#$ -cwd                    
#$ -l h_rt=8:00:00        
#$ -l h_vmem=16G           
#$ -q gpu                  # Request the GPU queue
#$ -l gpu=1                # Request exactly 1 GPU card
#$ -pe sharedmem 4        
#$ -m be                       # Email on begin and end
#$ -M s2190468@ed.ac.uk        # Your email address

# Initialize the module environment
. /etc/profile.d/modules.sh

# Load the newest version of MATLAB available on Eddie
module load matlab/R2024a 

matlab -nodisplay -nosplash -nodesktop -r "run('test3.m'); exit;"