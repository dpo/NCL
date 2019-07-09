using Ipopt

using NLPModels
using NLPModelsIpopt
using CUTEst
using SolverTools

using LinearAlgebra
using SparseArrays
using Test

using NCL

include("test_NCLSolve.jl")
include("test_NCLModel.jl")

"""
##########################
# Run every test for NCL #
##########################
"""
function test_main(test_NCLModel_command::Bool, test_NCLSolve_command::Bool) ::Test.DefaultTestSet
    test_NLCModel(test_NCLModel_command)
    test_NCLSolve(test_NCLSolve_command)
end
################################
################################
test_main(true,true)
