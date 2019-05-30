using Test
using NLPModels
using Ipopt 
using NLPModelsIpopt

include("../src/ncl.jl")

function test_ncl(test::Bool) #::Test.DefaultTestSet
    # Test problem
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
        nlc = NLCModel(nlp, y, ρ)::NLCModel

    resolution = ipopt(nlp, print_level = 0, tol = 0.01)
    x = resolution.solution
    #@show x
    #@show cons(nlp, x)
    # Get multipliers
    λ = resolution.solver_specific[:multipliers_con]
    #@show λ
    z_U = resolution.solver_specific[:multipliers_U]
    #@show z_U
    z_L = resolution.solver_specific[:multipliers_L]
    #@show z_L
#! Attention aux mult d'IPOPT !


    if test
        @testset "NCL algorithm" begin
            @testset "NLPModel_solved() function" begin
                @test NLPModel_solved(nlp, [0.5, 1.0], [-1, 0, 0, 2], [0, -1], [0, 0], 0.01, true) == true # solved by hand
                @test NLPModel_solved(nlp, [1.0, 0.5], -[0, 0, -1/3, -2/3], [1, 0], [0, 0], 0.01, true) == true # solved by hand
                @test NLPModel_solved(nlp, [0.5, 1.0], λ, z_U, z_L, 1, true) == true
            end
        end
    end
end