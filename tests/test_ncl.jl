using Test
using NLPModels
using Ipopt 
using NLPModelsIpopt

include("../src/ncl.jl")
#include("../src/NLPModelsIpopt_perso.jl")

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
                -0.5,
                0.5]
        c(x) = [x[1] - x[2], # linear
                x[1]^2 + x[2], # non linear one, range constraint
                x[1] - x[2], # linear, lower bounded 
                x[1] * x[2]] # equality one

        name = "Unitary test problem"
        nlp = ADNLPModel(f, x0; lvar=lvar, uvar=uvar, c=c, lcon=lcon, ucon=ucon, name=name)::ADNLPModel
        nlc = NLCModel(nlp, y, ρ)::NLCModel

    resolution_k = ipopt(nlp, print_level = 0, tol = 0.01)
    x_k = resolution_k.solution
    @show x_k
    println(cons(nlp, x_k))
    # Get multipliers
    y_k = resolution_k.solver_specific[:multipliers_con]
    @show y_k
    z_k_U = resolution_k.solver_specific[:multipliers_U]
    @show z_k_U
    z_k_L = resolution_k.solver_specific[:multipliers_L]
    @show z_k_L
#! Attention aux mult d'IPOPT !


    if test
        @testset "NCL algorithm" begin
            @testset "NLPModel_solved() function" begin
                @test NLPModel_solved(nlp, [0.5, 1.0], [1, 0, 0, -2], [0, -1], [0, 0], 1) == true
                @test NLPModel_solved(nlp, [0.5, 1.0], [-0.184774, 0, 0.621765, -1.43699], [5.01181e-9, 0.155487], [5.01181e-9, 2.5059e-9], 10) == true
            end
        end
    end
end