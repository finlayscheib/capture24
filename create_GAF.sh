#!/bin/sh
set -e

#$ -N GAF_images      
#$ -cwd                     # Run in the current working directory
#$ -l h_rt=20:00:00         
#$ -l h_vmem=16G            # Request 16GB of RAM
#$ -pe sharedmem 4  
#$ -M s2190468@ed.ac.uk     # 
#$ -m be  

. /etc/profile.d/modules.sh                  
module load anaconda
source activate benchmark
export MPLCONFIGDIR="/exports/eddie/scratch/s2190468/matplotlib_cache"
python create_GAF.py