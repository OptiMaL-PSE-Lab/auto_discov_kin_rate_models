#!/bin/bash
#PBS -N RUN_DNO_20
#PBS -o run_DNO_20.out
#PBS -e run_DNO_20.err
#PBS -lwalltime=48:00:0
#PBS -lselect=1:ncpus=8:mem=64g

module load julia/1.6.4

julia /rds/general/user/md1621/home/SymbolicRegression.jl/Decomposition_of_Nitrous_Oxide.jl
