#!/bin/bash -x

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=100:00:00

# set the number of tasks (processes) per node.
#SBATCH --ntasks-per-node=16

# set partition
#SBATCH --partition=BIOISI

# set name of job
#SBATCH --job-name=ASD_Pred

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=NONE

# send mail to this address
#SBATCH --mail-user=joaopinacio96@gmail.com

# run the application
echo "Now on: $PWD"
$HOME/miniconda3/bin/conda run -n dna python /work/joaoinacio/work/Tese_ASD_Gene_Pred/2_Emb_seq_maker/dnaBert/Embeddings_max.py
echo "Finished"
