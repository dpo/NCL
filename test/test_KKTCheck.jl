"""
Unit tests for `KKTCheck.jl`
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

    if test
        @testset "KKTCheck function on HS problems" begin
            for name in probs_KKT # several tests
                hs = CUTEstModel(name)
                test_name = name * " problem resolution"

                @testset "$test_name optimality via ipopt" begin

                    resol = NLPModelsIpopt.ipopt(hs, print_level=0)

                    if (name == "HS13") | (name == "HS55")
                        D = KKTCheck(hs, resol.solution, - resol.solver_specific[:multipliers_con] , resol.solver_specific[:multipliers_U] , resol.solver_specific[:multipliers_L])
                        @test_broken D[:optimal]
                        @test_broken D[:acceptable]
                    else
                        D = KKTCheck(hs, resol.solution, - resol.solver_specific[:multipliers_con] , resol.solver_specific[:multipliers_U] , resol.solver_specific[:multipliers_L])
                        @test D[:optimal]
                        @test D[:acceptable]
                    end

                end
                finalize(hs)
            end
        end

        @testset "KKTCheck(nlp) via ipopt" begin
            # Solution of NLP with NLPModelsIpopt
            resol_nlp_ipopt = NLPModelsIpopt.ipopt(nlp ; print_level = 0, tol = ω, constr_viol_tol = η, compl_inf_tol = ϵ)
            x_nlp_ipopt = resol_nlp_ipopt.solution

            # Get multipliers
            λ_nlp_ipopt = - resol_nlp_ipopt.solver_specific[:multipliers_con]
            z_U_nlp_ipopt = resol_nlp_ipopt.solver_specific[:multipliers_U]
            z_L_nlp_ipopt = resol_nlp_ipopt.solver_specific[:multipliers_L]

            D = KKTCheck(nlp, x_nlp_ipopt, λ_nlp_ipopt, z_U_nlp_ipopt, z_L_nlp_ipopt)
            @test D[:optimal]
            @test D[:acceptable]
        end
    else
        @testset "Empty test" begin
            @test true
        end
    end
end