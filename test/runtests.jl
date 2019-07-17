using Ipopt

using NLPModels
using NLPModelsIpopt
using CUTEst
using SolverTools

using LinearAlgebra
using SparseArrays
using Test

#using NCL

include("../src/NCLModel.jl")
include("test_NCLModel.jl")

include("../src/KKTCheck.jl")
include("test_KKTCheck.jl")

include("../src/NCLSolve.jl")
include("test_NCLSolve.jl")


"""
##########################
# Run every test for NCL #
##########################
"""
function test_main(test_NCLModel_command::Bool, test_KKTCheck_command::Bool, test_NCLSolve_command::Bool) ::Test.DefaultTestSet
    test_NCLModel(test_NCLModel_command)
    test_KKTCheck(test_KKTCheck_command)
    test_NCLSolve(test_NCLSolve_command)
end
################################
################################

test_main(true,true,true)
