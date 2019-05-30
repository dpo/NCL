
using Test
using NLPModels

include("test_ncl.jl")
include("test_NLCModel.jl")
include("../src/main.jl")

function test_main(test_NCLModel_command::Bool, test_ncl_command::Bool, test_main_command::Bool) ::Test.DefaultTestSet
    if test_NCLModel_command
        test_NLCModel(true)
    end
    if test_ncl_command
        test_ncl(true)
    end
    if test_main_command
        ρ = 1.
        y = [2., 1.]

        f(x) = x[1] + x[2]
        x0 = [1, 0.5]
        lvar = [0., 0.]
        uvar = [1., 1.]

        lcon = [-0.5,
                -1.,
                -Inf,
                0.5]
        ucon = [Inf,
                2.,
                0.5,
                0.5]
        c(x) = [x[1] - x[2], # linear
                x[1]^2 + x[2], # non linear one, range constraint
                x[1] - x[2], # linear, lower bounded 
                x[1] * x[2]] # equality one

        name = "Unitary test problem"
        nlp = ADNLPModel(f, x0; lvar=lvar, uvar=uvar, c=c, lcon=lcon, ucon=ucon, name=name)::ADNLPModel

        @testset "NCLMain" begin
            @test isa(NCLMain(nlp), GenericExecutionStats)
            @test NCLMain(nlp, max_iter = 15).iter <= 15
        end
    end
end

test_main(true,true,true)