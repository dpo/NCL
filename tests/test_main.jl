
using Test
using NLPModels
using CUTEst


include("test_NCLSolve.jl")
include("test_NCLModel.jl")

"""
############################################
# Little fonction to do every test for NCL #
############################################
"""
function test_main(test_NCLModel_command::Bool, test_NCLSolve_command::Bool) ::Test.DefaultTestSet
    test_NLCModel(test_NCLModel_command)
    test_NCLSolve(test_NCLSolve_command)
end
################################
################################
test_main(true,true)