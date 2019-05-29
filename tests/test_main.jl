
using Test
using NLPModels

include("test_ncl.jl")
include("test_NLCModel.jl")

function test_main(test_ncl_command::Bool, test_NCLModel_command::Bool, test_main_command::Bool) ::Test.DefaultTestSet
    test_NLCModel(test_NCLModel_command)
    test_ncl(test_ncl_command)
    @testset "Main NCL" begin
        @test true
    end
end

test_main(true,true,true)