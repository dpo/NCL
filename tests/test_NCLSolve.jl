using Test
using NLPModels
using Ipopt 
using NLPModelsIpopt

include("../src/NCLSolve.jl")
include("../src/NCLModel.jl")
probs = ["HS" * string(i) for i in 1:57]

function test_NCLSolve(test::Bool) ::Test.DefaultTestSet
    
    print_level = 0
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
        nlp = ADNLPModel(f, x0; lvar=lvar, uvar=uvar, c=c, lcon=lcon, ucon=ucon, name=name, lin = [1,3])::ADNLPModel
        ncl_nlin_res = NCLModel(nlp, res_lin_cons = false)::NCLModel

        ncl_nlin_res.y = y
        ncl_nlin_res.ρ = ρ

        nlc_cons_res = NCLModel(nlp, res_lin_cons = true)::NCLModel

    if test
        @testset "KKT_check function" begin
            for name in probs # several tests
                hs = CUTEstModel(name)
                test_name = name * " problem resolution"
                @testset "$test_name optimality via ipopt" begin
                    resol = NLPModelsIpopt.ipopt(hs, max_iter = 5000, print_level=0, tol = ω, constr_viol_tol = η, compl_inf_tol = ϵ)
                    @test KKT_check(hs, resol.solution, - resol.solver_specific[:multipliers_con] , resol.solver_specific[:multipliers_U] , resol.solver_specific[:multipliers_L] , ω, η, ϵ, 7)
                end
                finalize(hs)
            end
        end


        @testset "NCLSolve.jl (only linear residuals)" begin

            @testset "KKT_check function" begin
                @testset "KKT_check(nlp) via ipopt" begin
                    # Resolution of NLP with NLPModelsIpopt
                        resol_nlp_ipopt = NLPModelsIpopt.ipopt(nlp, print_level = 0, tol = ω, constr_viol_tol = η, compl_inf_tol = ϵ, ignore_time = true)
                        x_nlp_ipopt = resol_nlp_ipopt.solution
                        
                        # Get multipliers
                        λ_nlp_ipopt = - resol_nlp_ipopt.solver_specific[:multipliers_con]
                        z_U_nlp_ipopt = resol_nlp_ipopt.solver_specific[:multipliers_U]
                        z_L_nlp_ipopt = resol_nlp_ipopt.solver_specific[:multipliers_L]

                    @test_broken KKT_check(nlp, [0.5, 1.0], [1., 0., 0., -2.0], [0, 1.], [0., 0.0], ω, η, ϵ, print_level) # solved by hand
                    @test KKT_check(nlp, x_nlp_ipopt, λ_nlp_ipopt, z_U_nlp_ipopt, z_L_nlp_ipopt, ω, η, ϵ, print_level)
                end

                @testset "KKT_check(ncl_nlin_res) via ipopt" begin
                    # Resolution of ncl_nlin_res with NLPModelsIpopt
                        resol_ncl_ipopt = NLPModelsIpopt.ipopt(ncl_nlin_res, print_level = 0, tol = ω, constr_viol_tol = η, compl_inf_tol = ϵ, ignore_time = true)
                        x_ncl_ipopt = resol_ncl_ipopt.solution
                        
                        # Get multipliers
                        λ_ncl_ipopt = - resol_ncl_ipopt.solver_specific[:multipliers_con]
                        z_U_ncl_ipopt = resol_ncl_ipopt.solver_specific[:multipliers_U]
                        z_L_ncl_ipopt = resol_ncl_ipopt.solver_specific[:multipliers_L]

                    @test KKT_check(ncl_nlin_res, x_ncl_ipopt, λ_ncl_ipopt, z_U_ncl_ipopt, z_L_ncl_ipopt, ω, η, ϵ, print_level)
                end

                
            end

            @testset "KKT_check(nlp) via NCLSolve" begin
                # Resolution of ncl_nlin_res with NCL method
                    resol_ncl_ncl = NCLSolve(ncl_nlin_res, max_iter_NCL = 30, use_ipopt = true, tol = ω, constr_viol_tol = η, compl_inf_tol = ϵ, print_level = print_level)
                    x_ncl = resol_ncl_ncl.solution

                    λ_ncl = resol_ncl_ncl.solver_specific[:multipliers_con]
                    z_U_ncl = resol_ncl_ncl.solver_specific[:multipliers_U]
                    z_L_ncl = resol_ncl_ncl.solver_specific[:multipliers_L]    
                
                @test KKT_check(nlp, x_ncl[1:ncl_nlin_res.nvar_x], λ_ncl, z_U_ncl[1:ncl_nlin_res.nvar_x], z_L_ncl[1:ncl_nlin_res.nvar_x], ω, η, ϵ, print_level) 
            end
        end


        @testset "NCLSolve.jl (all residuals)" begin
            @testset "KKT_check(nlc_cons_res) via ipopt" begin
                # Resolution of ncl_nlin_res with NLPModelsIpopt
                    resol_ncl_ipopt = NLPModelsIpopt.ipopt(nlc_cons_res, print_level = 0, tol = ω, constr_viol_tol = η, compl_inf_tol = ϵ, ignore_time = true)
                    x_ncl_ipopt = resol_ncl_ipopt.solution
                    
                    # Get multipliers
                    λ_ncl_ipopt = - resol_ncl_ipopt.solver_specific[:multipliers_con]
                    z_U_ncl_ipopt = resol_ncl_ipopt.solver_specific[:multipliers_U]
                    z_L_ncl_ipopt = resol_ncl_ipopt.solver_specific[:multipliers_L]

                @test KKT_check(nlc_cons_res, x_ncl_ipopt, λ_ncl_ipopt, z_U_ncl_ipopt, z_L_ncl_ipopt, ω, η, ϵ, print_level)
            end

            @testset "KKT_check(nlp) via NCLSolve" begin
                # Resolution of nlc_cons_res with NCL method
                    resol_ncl_ncl = NCLSolve(nlc_cons_res, max_iter_NCL = 30, use_ipopt = true, tol = ω, constr_viol_tol = η, compl_inf_tol = ϵ, print_level = print_level)
                    x_ncl = resol_ncl_ncl.solution
            
                    λ_ncl = resol_ncl_ncl.solver_specific[:multipliers_con]
                    z_U_ncl = resol_ncl_ncl.solver_specific[:multipliers_U]
                    z_L_ncl = resol_ncl_ncl.solver_specific[:multipliers_L]

                @test KKT_check(nlp, x_ncl[1:nlc_cons_res.nvar_x], λ_ncl, z_U_ncl[1:nlc_cons_res.nvar_x], z_L_ncl[1:nlc_cons_res.nvar_x], ω, η, ϵ, print_level) 
            end
        end
    else
        @testset "Avoid return type bug" begin
            @test true
        end
    end
end