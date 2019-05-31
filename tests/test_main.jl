
using Test
using NLPModels
using CUTEst

function decodemodel(name)
    finalize(name)
    CUTEstModel(name)
end

probs = ["AKIVA", "ALLINITU", "ARGLINA", "ARGLINB", "ARGLINC","ARGTRIGLS", "ARWHEAD"]
broadcast(decodemodel, probs)

addprocs(2)
@everywhere using CUTEst
@everywhere function evalmodel(name)
   nlp = CUTEstModel(name; decode=false)
   retval = obj(nlp, nlp.meta.x0)
   finalize(nlp)
   retval
end

fvals = pmap(evalmodel, probs)






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

        name_nlp = "Unitary test problem"
        nlp = ADNLPModel(f, x0; lvar=lvar, uvar=uvar, c=c, lcon=lcon, ucon=ucon, name=name_nlp)::ADNLPModel

        @testset "NCLMain" begin
            @test isa(NCLMain(nlp), Tuple{GenericExecutionStats, Bool})
            @test NCLMain(nlp, max_iter = 15)[1].iter <= 15
        end

        @testset "Problem resolution" for name in problem_names
            @test NCLMain(CUTEstModel(name; decode=false), max_iter = 100)[2] # several tests
        end
    end
end

test_main(true,true,true)