using NCL
using Test
using NLPModels
using CUTEst
using SolverTools
using NLPModelsIpopt


include("test_NCLModel.jl")
include("test_KKTCheck.jl")
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
