using Test
using NLPModels
using Ipopt 
using NLPModelsIpopt

include("../src/ncl.jl")

function test_ncl(test::Bool) ::Test.DefaultTestSet
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
        nlc = NLCModel(nlp)::NLCModel

    nlc.y = y
    nlc.ρ = ρ

    # Resolution of NLP with NLPModelsIpopt #! Attention aux mult d'IPOPT !
        resolution_nlp_ipopt = ipopt(nlp, print_level = 0, tol = 0.01)
        x_nlp_ipopt = resolution_nlp_ipopt.solution
        
        # Get multipliers
        λ_nlp_ipopt = resolution_nlp_ipopt.solver_specific[:multipliers_con]
        z_U_nlp_ipopt = resolution_nlp_ipopt.solver_specific[:multipliers_U]
        z_L_nlp_ipopt = resolution_nlp_ipopt.solver_specific[:multipliers_L]

    # Resolution of NLC with NLPModelsIpopt
        resolution_nlc_ipopt = ipopt(nlc, print_level = 0, tol = 0.01)
        x_nlc_ipopt = resolution_nlc_ipopt.solution
        
        # Get multipliers
        λ_nlc_ipopt = resolution_nlc_ipopt.solver_specific[:multipliers_con]
        z_U_nlc_ipopt = resolution_nlc_ipopt.solver_specific[:multipliers_U]
        z_L_nlc_ipopt = resolution_nlc_ipopt.solver_specific[:multipliers_L]
    

    
    # Resolution of NLC with NCL method
        printing_check = false
        printing_iterations = false
        printing_iterations_solver = false

        resolution_nlc_ncl = ncl(nlc, 50, true, 0.1, printing_iterations, printing_iterations_solver, printing_check)
        x_ncl = resolution_nlc_ncl.solution
        λ_ncl = resolution_nlc_ncl.solver_specific[:multipliers_con]
        z_U_ncl = resolution_nlc_ncl.solver_specific[:multipliers_U]
        z_L_ncl = resolution_nlc_ncl.solver_specific[:multipliers_L]


    
    if test
        @testset "ncl.jl" begin
            #! NLPModel_solved doesn't work every time, probably because of sign of multipliers (with sign of constraint and jacobian)...
            # TODO: fix this problem...

            @testset "NLPModel_solved(nlp) function" begin
                @test NLPModel_solved(nlp, [0.5, 1.0], [-1, 0, 0, 2], [0, -1], [0, 0], 0.01, printing_check) # solved by hand
                @test NLPModel_solved(nlp, x_nlp_ipopt, -λ_nlp_ipopt, z_U_nlp_ipopt, z_L_nlp_ipopt, 1, printing_check)
            end

            @testset "NLPModel_solved(nlc) function" begin
                @test_broken NLPModel_solved(nlc, x_nlc_ipopt, -λ_nlc_ipopt, z_U_nlc_ipopt, z_L_nlc_ipopt, 1, printing_check)
            end

            @testset "ncl algorithm" begin
                @test NLPModel_solved(nlp, x_ncl[1:nlc.nvar_x], -λ_ncl, z_U_ncl[1:nlc.nvar_x], z_L_ncl[1:nlc.nvar_x], 0.1, printing_check) 
            end

        end
    end
end