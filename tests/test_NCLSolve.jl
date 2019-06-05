using Test
using NLPModels
using Ipopt 
using NLPModelsIpopt

include("../src/NCLSolve.jl")

function test_NCLSolve(test::Bool) ::Test.DefaultTestSet
    printing = false
    printing_check = printing
    printing_iterations = printing
    printing_iterations_solver = false
    ω = 0.001
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
        ncl = NCLModel(nlp)::NCLModel

        ncl.y = y
        ncl.ρ = ρ


    if test

        # Resolution of NLP with NLPModelsIpopt #! Attention aux mult d'IPOPT !
            resol_nlp_ipopt = NLPModelsIpopt.ipopt(nlp, print_level = 0, tol = ω, constr_viol_tol = η, compl_inf_tol = ϵ, ignore_time = true)
            x_nlp_ipopt = resol_nlp_ipopt.solution
            
            # Get multipliers
            λ_nlp_ipopt = - resol_nlp_ipopt.solver_specific[:multipliers_con]
            z_U_nlp_ipopt = resol_nlp_ipopt.solver_specific[:multipliers_U]
            z_L_nlp_ipopt = resol_nlp_ipopt.solver_specific[:multipliers_L]

        # Resolution of NCL with NLPModelsIpopt
            resol_ncl_ipopt = NLPModelsIpopt.ipopt(ncl, print_level = 0, tol = ω, constr_viol_tol = η, compl_inf_tol = ϵ, ignore_time = true)
            x_ncl_ipopt = resol_ncl_ipopt.solution
            
            # Get multipliers
            λ_ncl_ipopt = -resol_ncl_ipopt.solver_specific[:multipliers_con]
            z_U_ncl_ipopt = resol_ncl_ipopt.solver_specific[:multipliers_U]
            z_L_ncl_ipopt = resol_ncl_ipopt.solver_specific[:multipliers_L]
        
            
        
        # Resolution of NCL with NCL method
            resol_ncl_ncl = NCLSolve(ncl, max_iter = 10, use_ipopt = true, tol = ω, constr_viol_tol = η, compl_inf_tol = ϵ, printing_iterations=printing_iterations, printing_iterations_solver=false, printing_check=printing_check)
            y_end = copy(ncl.y)
            x_ncl = resol_ncl_ncl.solution

            λ_ncl = resol_ncl_ncl.solver_specific[:multipliers_con]
            z_U_ncl = resol_ncl_ncl.solver_specific[:multipliers_U]
            z_L_ncl = resol_ncl_ncl.solver_specific[:multipliers_L]




        @testset "NCLSolve.jl" begin

            @testset "KKT_check function" begin
                @testset "KKT_check(nlp)" begin
                    @test_broken KKT_check(nlp, [0.5, 1.0], [1., 0., 0., -2.0], [0, 1.], [0., 0.0], ω, η, ϵ, printing_check) # solved by hand
                    @test KKT_check(nlp, x_nlp_ipopt, λ_nlp_ipopt, z_U_nlp_ipopt, z_L_nlp_ipopt, ω, η, ϵ, printing_check)
                end

                @testset "NCLSolve algorithm to resolve nlp" begin
                    @test KKT_check(nlp, x_ncl[1:ncl.nvar_x], λ_ncl, z_U_ncl[1:ncl.nvar_x], z_L_ncl[1:ncl.nvar_x], ω, η, ϵ, printing_check) 
                end

                @testset "KKT_check(ncl)" begin
                    ncl.y = [2., 1.]
                    ncl.ρ = ρ # back to the first value (it was modified by NCLSolve)
                    @test KKT_check(ncl, x_ncl_ipopt, λ_ncl_ipopt, z_U_ncl_ipopt, z_L_ncl_ipopt, ω, η, ϵ, printing_check) # Complémentarité 2eme contrainte non respectée
                end
            end
        end
    else
        @testset "Avoid return type bug" begin
            @test true
        end
    end
end