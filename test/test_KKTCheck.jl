#include("../src/NCLModel.jl")
#include("../src/KKTCheck.jl")
#include("../src/NCLSolve.jl")


"""
##############################
# Unit tests for NCLSolve.jl #
##############################
"""
function test_KKTCheck(test::Bool ; HS_begin_KKT::Int64 = 1, HS_end_KKT::Int64 = 8) ::Test.DefaultTestSet
    # Test parameters
    print_level_NCL = 0
    ω = 0.001
    η = 0.0001
    ϵ = 0.0001
    probs_KKT = ["HS" * string(i) for i in HS_begin_KKT:HS_end_KKT]

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
    c(x) = [x[1] - x[2],   # linear
            x[1]^2 + x[2], # nonlinear range constraint
            x[1] - x[2],   # linear, lower bounded
            x[1] * x[2]]   # equality constraint

    name = "Unit test problem"
    nlp = ADNLPModel(f, x0; lvar=lvar, uvar=uvar, c=c, lcon=lcon, ucon=ucon, name=name, lin = [1,3])::ADNLPModel
    ncl_nlin_res = NCLModel(nlp ; res_lin_cons = false)::NCLModel

    ncl_nlin_res.y = y
    ncl_nlin_res.ρ = ρ

    nlc_cons_res = NCLModel(nlp ; res_lin_cons = true)::NCLModel


    @testset "KKTCheck function" begin
        for name in probs_KKT # several tests
            hs = CUTEstModel(name)
            test_name = name * " problem resolution"

            @testset "$test_name optimality via ipopt" begin

                resol = NLPModelsIpopt.ipopt(hs, print_level=0)

                if (name == "HS13") | (name == "HS55")
                    D = KKTCheck(hs, resol.solution, - resol.solver_specific[:multipliers_con] , resol.solver_specific[:multipliers_U] , resol.solver_specific[:multipliers_L])
                    @test_broken D["optimal"]
                    @test_broken D["acceptable"]
                else
                    D = KKTCheck(hs, resol.solution, - resol.solver_specific[:multipliers_con] , resol.solver_specific[:multipliers_U] , resol.solver_specific[:multipliers_L])
                    @test D["optimal"]
                    @test D["acceptable"]
                end

            end
            finalize(hs)
        end
    end

    @testset "KKTCheck function" begin
        @testset "KKTCheck(nlp) via ipopt" begin
            # Solution of NLP with NLPModelsIpopt
            resol_nlp_ipopt = NLPModelsIpopt.ipopt(nlp ; print_level = 0, tol = ω, constr_viol_tol = η, compl_inf_tol = ϵ, ignore_time = true)
            x_nlp_ipopt = resol_nlp_ipopt.solution

            # Get multipliers
            λ_nlp_ipopt = - resol_nlp_ipopt.solver_specific[:multipliers_con]
            z_U_nlp_ipopt = resol_nlp_ipopt.solver_specific[:multipliers_U]
            z_L_nlp_ipopt = resol_nlp_ipopt.solver_specific[:multipliers_L]

            D = KKTCheck(nlp, x_nlp_ipopt, λ_nlp_ipopt, z_U_nlp_ipopt, z_L_nlp_ipopt)
            @test D["optimal"]
            @test D["acceptable"]
        end

        @testset "KKTCheck(ncl_nlin_res) via ipopt" begin
            # Solution of ncl_nlin_res with NLPModelsIpopt
            resol_ncl_ipopt = NLPModelsIpopt.ipopt(ncl_nlin_res ; print_level = 0, tol = ω, constr_viol_tol = η, compl_inf_tol = ϵ, ignore_time = true)
            x_ncl_ipopt = resol_ncl_ipopt.solution

            # Get multipliers
            λ_ncl_ipopt = - resol_ncl_ipopt.solver_specific[:multipliers_con]
            z_U_ncl_ipopt = resol_ncl_ipopt.solver_specific[:multipliers_U]
            z_L_ncl_ipopt = resol_ncl_ipopt.solver_specific[:multipliers_L]

            D = KKTCheck(ncl_nlin_res, x_ncl_ipopt, λ_ncl_ipopt, z_U_ncl_ipopt, z_L_ncl_ipopt)
            @test D["optimal"]
            @test D["acceptable"]
        end
    end
end