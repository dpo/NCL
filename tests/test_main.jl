
using Test
using NLPModels

include("test_ncl.jl")
include("test_NLCModel.jl")

function test_main(test_NCLModel_command::Bool, test_ncl_command::Bool, test_main_command::Bool) #::Test.DefaultTestSet
    if test_NCLModel_command
        test_NLCModel(true)
    end
    if test_ncl_command
        test_ncl(false)
    end
    if test_main_command
        @testset "Main NCL" begin
            @test true
        end
    end
end

test_main(true,true,false)