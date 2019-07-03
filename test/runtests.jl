import Pkg
Pkg.add("NLPModels")
Pkg.add("CUTEst")
Pkg.add("Ipopt")
Pkg.add("NLPModelsIpopt")
Pkg.add("SolverTools")

Pkg.add("LinearAlgebra")
Pkg.add("SparseArrays")

Pkg.add("Test")

using NLPModels, CUTEst, Ipopt, NLPModelsIpopt, SolverTools, LinearAlgebra, SparseArrays, Test

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
