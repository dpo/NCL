
using Test
using NLPModels

include("test_ncl.jl")
include("test_NLCModel.jl")

function test_main(test_ncl::Bool, test_NCLModel::Bool)
    test_NCLModel(test_NCLModel)
    test_ncl(test_ncl)
end