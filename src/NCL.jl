module NCL

# comment
#** Important
# ! Warning / Problem
# ? Question
# TODO

# Solver
using Ipopt

# Julia packages
using LinearAlgebra
using Printf
using SparseArrays

# Problem modelisation and structures
using NLPModels
using NLPModelsIpopt
using SolverTools

# For pb_set benchmarks
using DataFrames
using Plots

using CUTEst
using AmplNLReader
using SolverBenchmark

# Module files
include("NCLModel.jl")
include("KKTCheck.jl")
include("NCLSolve.jl")
include("pb_set_resol.jl")


######### TODO #########
    # TODO (feature)   : Créer un vrai statut
    # TODO KKTCheck output to make uniform with NCLSolve
########## TODO ########


end #end of module
