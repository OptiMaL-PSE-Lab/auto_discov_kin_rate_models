#!/bin/bash
#PBS -N RUN_T2B
#PBS -o run_T2B.out
#PBS -e run_T2B.err
#PBS -lwalltime=48:00:0
#PBS -lselect=1:ncpus=8:mem=64g

module load julia/1.6.4

julia /rds/general/user/md1621/home/SymbolicRegression.jl/Hydrodealkylation_of_Toluene.jl
