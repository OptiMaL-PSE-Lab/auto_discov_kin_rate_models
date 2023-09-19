#!/bin/bash
#PBS -N RUN_SI_10
#PBS -o run_SI_10.out
#PBS -e run_SI_10.err
#PBS -lwalltime=48:00:0
#PBS -lselect=1:ncpus=8:mem=64g

module load julia/1.6.4

julia /rds/general/user/md1621/home/SymbolicRegression.jl/Synthetic_Isomerisation.jl
