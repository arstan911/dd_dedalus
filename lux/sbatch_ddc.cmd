#!/bin/bash
#SBATCH --job-name="DD_dedalus_001" # Job name
#SBATCH --partition=cpuq
#SBATCH --account=cpuq
#SBATCH --output="DD_dedalus_001.o%j" # Name of stdout output file
#SBATCH --error="DD_dedalus_001.e%j" # Name of stderr error file
#SBATCH --nodes=1 # Total number of nodes
#SBATCH --ntasks=40
#SBATCH --ntasks-per-node=40 # Total number of mpi tasks per node
#SBATCH -t 00:45:00 # Run time (hh:mm:ss)
module load  openmpi/4.0.1 fftw/3.3.8 python/3.8.6
export OMP_NUM_THREADS=1
mpiexec -n 40 python3 double_diffusive_fixed_flux.py
mpiexec -n 40 python3 -m dedalus merge_procs snapshots
mpiexec -n 40 python3 -m dedalus merge_procs scalars
mpiexec -n 40 python3 -m dedalus merge_procs vertical_profiles
mpiexec -n 40 python3 plot_slices.py
