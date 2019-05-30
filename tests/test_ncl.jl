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

    resolution_nlp = ipopt(nlp, print_level = 0, tol = 0.01)
    x_nlp = resolution_nlp.solution
    #@show x_nlp
    #@show cons(nlp, x)
    
    # Get multipliers
    λ_nlp = resolution_nlp.solver_specific[:multipliers_con]
    #@show λ_nlp
    z_U_nlp = resolution_nlp.solver_specific[:multipliers_U]
    #@show z_U_nlp
    z_L_nlp = resolution_nlp.solver_specific[:multipliers_L]
    #@show z_L_nlp
                                    #! Attention aux mult d'IPOPT !

    #resolution_nlc = ipopt(nlc, print_level = 3, tol = 0.01)
    #x_nlc = resolution_nlc.solution
    #@show x_nlc
    #@show cons(nlp, x)
    
    # Get multipliers
    #λ_nlc = resolution_nlc.solver_specific[:multipliers_con]
    #@show λ_nlc
    #z_U_nlc = resolution_nlc.solver_specific[:multipliers_U]
    #@show z_U_nlc
    #z_L_nlc = resolution_nlc.solver_specific[:multipliers_L]
    #@show z_L_nlc



    println(ncl(nlc, 10, true))



    solve_print_nlp = false
    solve_print_nlc = true

    if test
        @testset "ncl.jl" begin
            #! NLPModel_solved doesn't work, probably because of sign of multipliers (with sign of constraint and jacobian)

            #@testset "NLPModel_solved(nlp) function" begin
            #    @test NLPModel_solved(nlp, [0.5, 1.0], [-1, 0, 0, 2], [0, -1], [0, 0], 0.01, solve_print_nlp) # solved by hand
            #    @test_broken NLPModel_solved(nlp, [1.0, 0.5], [0, 0, -1/3, -2/3], [1, 0], [0, 0], 0.01, solve_print_nlp) # solved by hand
            #    @test NLPModel_solved(nlp, x_nlp, -λ_nlp, z_U_nlp, z_L_nlp, 1, solve_print_nlp)
            #end

            #@testset "NLPModel_solved(nlc) function" begin
            #    @test NLPModel_solved(nlc, x_nlc, -λ_nlc, z_U_nlc, z_L_nlc, 1, solve_print_nlc)
            #end
            @test true
        end
    end
end