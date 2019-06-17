using Test
using NLPModels
using Ipopt 
using NLPModelsIpopt

include("../src/NCLSolve.jl")
include("../src/NCLModel.jl")





"""
#############################
Unitary tests for NCLSolve.jl
#############################
"""
function test_NCLSolve(test::Bool ; HS_begin_KKT::Int64 = 1, HS_end_KKT::Int64 = 13, HS_begin_NCL::Int64 = 1,  HS_end_NCL::Int64 = 13) ::Test.DefaultTestSet
    # Test parameters
        print_level_NCL = 0
        ω = 0.001
        η = 0.0001
        ϵ = 0.0001
        probs_KKT = ["HS" * string(i) for i in HS_begin_KKT:HS_end_KKT]
        probs_NCL = ["HS" * string(i) for i in HS_begin_NCL:HS_end_NCL] #[13,15,17,19,20]

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
        ncl_nlin_res = NCLModel(nlp ; res_lin_cons = false)::NCLModel

        ncl_nlin_res.y = y
        ncl_nlin_res.ρ = ρ

        nlc_cons_res = NCLModel(nlp ; res_lin_cons = true)::NCLModel

    if test
        @testset "KKT_check function" begin
            for name in probs_KKT # several tests
                hs = CUTEstModel(name)
                test_name = name * " problem resolution"

                @testset "$test_name optimality via ipopt" begin
                    
                    resol = NLPModelsIpopt.ipopt(hs, print_level=0)
                    
                    if (name == "HS13") | (name == "HS49") | (name == "HS55")
                        @test_broken KKT_check(hs, resol.solution, - resol.solver_specific[:multipliers_con] , resol.solver_specific[:multipliers_U] , resol.solver_specific[:multipliers_L])
                    else
                        @test KKT_check(hs, resol.solution, - resol.solver_specific[:multipliers_con] , resol.solver_specific[:multipliers_U] , resol.solver_specific[:multipliers_L])
                    end

                end
                finalize(hs)
            end
        end


        @testset "NCLSolve NLP (only linear residuals)" begin

            @testset "KKT_check function" begin
                @testset "KKT_check(nlp) via ipopt" begin
                    # Resolution of NLP with NLPModelsIpopt
                        resol_nlp_ipopt = NLPModelsIpopt.ipopt(nlp ; print_level = 0, tol = ω, constr_viol_tol = η, compl_inf_tol = ϵ, ignore_time = true)
                        x_nlp_ipopt = resol_nlp_ipopt.solution
                        
                        # Get multipliers
                        λ_nlp_ipopt = - resol_nlp_ipopt.solver_specific[:multipliers_con]
                        z_U_nlp_ipopt = resol_nlp_ipopt.solver_specific[:multipliers_U]
                        z_L_nlp_ipopt = resol_nlp_ipopt.solver_specific[:multipliers_L]

                    @test_broken KKT_check(nlp, [0.5, 1.0], [1., 0., 0., -2.0], [0, 1.], [0., 0.0]) # solved by hand
                    @test KKT_check(nlp, x_nlp_ipopt, λ_nlp_ipopt, z_U_nlp_ipopt, z_L_nlp_ipopt)
                end

                @testset "KKT_check(ncl_nlin_res) via ipopt" begin
                    # Resolution of ncl_nlin_res with NLPModelsIpopt
                        resol_ncl_ipopt = NLPModelsIpopt.ipopt(ncl_nlin_res ; print_level = 0, tol = ω, constr_viol_tol = η, compl_inf_tol = ϵ, ignore_time = true)
                        x_ncl_ipopt = resol_ncl_ipopt.solution
                        
                        # Get multipliers
                        λ_ncl_ipopt = - resol_ncl_ipopt.solver_specific[:multipliers_con]
                        z_U_ncl_ipopt = resol_ncl_ipopt.solver_specific[:multipliers_U]
                        z_L_ncl_ipopt = resol_ncl_ipopt.solver_specific[:multipliers_L]

                    @test KKT_check(ncl_nlin_res, x_ncl_ipopt, λ_ncl_ipopt, z_U_ncl_ipopt, z_L_ncl_ipopt)
                end

                
            end

            @testset "KKT_check(nlp) via NCLSolve" begin
                # Resolution of ncl_nlin_res with NCL method
                    resol_ncl_ncl = NCLSolve(ncl_nlin_res)
                    x_ncl = resol_ncl_ncl.solution

                    λ_ncl = resol_ncl_ncl.solver_specific[:multipliers_con]
                    z_U_ncl = resol_ncl_ncl.solver_specific[:multipliers_U]
                    z_L_ncl = resol_ncl_ncl.solver_specific[:multipliers_L]    
                
                @test KKT_check(nlp, x_ncl[1:ncl_nlin_res.nvar_x], λ_ncl, z_U_ncl[1:ncl_nlin_res.nvar_x], z_L_ncl[1:ncl_nlin_res.nvar_x]) 
            end
        end


        @testset "NCLSolve NLP (all residuals)" begin
            @testset "KKT_check(nlc_cons_res) via ipopt" begin
                # Resolution of ncl_nlin_res with NLPModelsIpopt
                    resol_ncl_ipopt = NLPModelsIpopt.ipopt(nlc_cons_res ; print_level = 0, tol = ω, constr_viol_tol = η, compl_inf_tol = ϵ, ignore_time = true)
                    x_ncl_ipopt = resol_ncl_ipopt.solution
                    
                    # Get multipliers
                    λ_ncl_ipopt = - resol_ncl_ipopt.solver_specific[:multipliers_con]
                    z_U_ncl_ipopt = resol_ncl_ipopt.solver_specific[:multipliers_U]
                    z_L_ncl_ipopt = resol_ncl_ipopt.solver_specific[:multipliers_L]

                @test KKT_check(nlc_cons_res, x_ncl_ipopt, λ_ncl_ipopt, z_U_ncl_ipopt, z_L_ncl_ipopt)
            end

            @testset "KKT_check(nlp) via NCLSolve" begin
                # Resolution of nlc_cons_res with NCL method
                    resol_ncl_ncl = NCLSolve(nlc_cons_res)
                    x_ncl = resol_ncl_ncl.solution
            
                    λ_ncl = resol_ncl_ncl.solver_specific[:multipliers_con]
                    z_U_ncl = resol_ncl_ncl.solver_specific[:multipliers_U]
                    z_L_ncl = resol_ncl_ncl.solver_specific[:multipliers_L]

                @test KKT_check(nlp, x_ncl[1:nlc_cons_res.nvar_x], λ_ncl, z_U_ncl[1:nlc_cons_res.nvar_x], z_L_ncl[1:nlc_cons_res.nvar_x]) 
            end
        end

        @testset "NCLSolve HS (only linear residuals)" begin
            for name in probs_NCL # several tests
                nlp = CUTEstModel(name)
                test_name = name * " problem resolution"
                @testset "$test_name" begin
                    @test NCLSolve(nlp ; linear_residuals=false).solver_specific[:internal_msg] == Symbol("Solve_Succeeded")
                end
                finalize(nlp)
            end
        end

        @testset "NCLSolve HS (all residuals)" begin
            for name in probs_NCL # several tests
                nlp = CUTEstModel(name)
                test_name = name * " problem resolution"
                @testset "$test_name" begin
                    @test NCLSolve(nlp ; linear_residuals = true).solver_specific[:internal_msg] == Symbol("Solve_Succeeded")
                end
                finalize(nlp)
            end
        end

    else
        @testset "Empty test" begin
            @test true
        end
    end
end
#############################
#############################