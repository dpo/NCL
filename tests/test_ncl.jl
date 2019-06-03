using Test
using NLPModels
using Ipopt 
using NLPModelsIpopt

include("../src/ncl.jl")

function test_ncl(test::Bool) ::Test.DefaultTestSet
    
    printing_check = true
    printing_iterations = true
    printing_iterations_solver = false
    ω = 0.01
    η = 0.0001
    ϵ = 0.0001
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
        resol_nlp_ipopt = NLPModelsIpopt.ipopt(nlp, print_level = 0, tol = ω, constr_viol_tol = η, compl_inf_tol = ϵ, ignore_time = true)
        x_nlp_ipopt = resol_nlp_ipopt.solution
        
        # Get multipliers
        λ_nlp_ipopt = resol_nlp_ipopt.solver_specific[:multipliers_con]
        z_U_nlp_ipopt = resol_nlp_ipopt.solver_specific[:multipliers_U]
        z_L_nlp_ipopt = resol_nlp_ipopt.solver_specific[:multipliers_L]

    # Resolution of NLC with NLPModelsIpopt
        resol_nlc_ipopt = NLPModelsIpopt.ipopt(nlc, print_level = 0, tol = ω, constr_viol_tol = η, compl_inf_tol = ϵ, ignore_time = true)
        x_nlc_ipopt = resol_nlc_ipopt.solution
        
        # Get multipliers
        λ_nlc_ipopt = resol_nlc_ipopt.solver_specific[:multipliers_con]
        z_U_nlc_ipopt = resol_nlc_ipopt.solver_specific[:multipliers_U]
        z_L_nlc_ipopt = resol_nlc_ipopt.solver_specific[:multipliers_L]
    
        @show x_nlc_ipopt
        @show λ_nlc_ipopt
        @show cons(nlc, x_nlc_ipopt)
    
    # Resolution of NLC with NCL method
        resol_nlc_ncl = NCLSolve(nlc, 50, true, ω, η, ϵ, printing_iterations, printing_iterations_solver, printing_check)
        x_ncl = resol_nlc_ncl.solution
        λ_ncl = resol_nlc_ncl.solver_specific[:multipliers_con]
        z_U_ncl = resol_nlc_ncl.solver_specific[:multipliers_U]
        z_L_ncl = resol_nlc_ncl.solver_specific[:multipliers_L]


    
    if test
        @testset "ncl.jl" begin
            #! NLPModel_solved doesn't work every time, probably because of sign of multipliers (with sign of constraint and jacobian)...
            # TODO: fix this problem...

            @testset "NLPModel_solved(nlp) function" begin
                @test NLPModel_solved(nlp, [0.5, 1.0], [-1, 0, 0, 2], [0, -1], [0, 0], ω, η, ϵ, printing_check) # solved by hand
                @test NLPModel_solved(nlp, x_nlp_ipopt, -λ_nlp_ipopt, z_U_nlp_ipopt, z_L_nlp_ipopt, ω, η, ϵ, printing_check)
            end

            @testset "NLPModel_solved(nlc) function" begin
                @test_broken NLPModel_solved(nlc, x_nlc_ipopt, -λ_nlc_ipopt, z_U_nlc_ipopt, z_L_nlc_ipopt, ω, η, ϵ, printing_check) # Complémentarité 2eme contrainte non respectée
            end

            @testset "ncl algorithm" begin
                @test NLPModel_solved(nlp, x_ncl[1:nlc.nvar_x], -λ_ncl, z_U_ncl[1:nlc.nvar_x], z_L_ncl[1:nlc.nvar_x], ω, η, ϵ, printing_check) 
            end

        end
    else
        @testset "Avoid type bug" begin
            @test true
        end
    end
end