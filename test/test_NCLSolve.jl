#include("../src/NCLSolve.jl")

"""
##############################
# Unit tests for NCLSolve.jl #
##############################
"""
function test_NCLSolve(test::Bool ; HS_begin_NCL::Int64 = 1,  HS_end_NCL::Int64 = 12) ::Test.DefaultTestSet
    # Test parameters
    print_level_NCL = 0
    ω = 0.001
    η = 0.0001
    ϵ = 0.0001

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
    c(x) = [x[1] - x[2],   # linear
            x[1]^2 + x[2], # nonlinear range constraint
            x[1] - x[2],   # linear, lower bounded
            x[1] * x[2]]   # equality constraint

    name = "Unit test problem"
    nlp = ADNLPModel(f, x0; lvar=lvar, uvar=uvar, c=c, lcon=lcon, ucon=ucon, name=name, lin = [1,3])::ADNLPModel

    nlc_cons_res = NCLModel(nlp)::NCLModel

    if test
        @test NCLSolve(nlp).solver_specific[:internal_msg] == Symbol("Solve_Succeeded")
        @test NCLSolve(nlc_cons_res).solver_specific[:internal_msg] == Symbol("Solve_Succeeded")

        @testset "NCLSolve HS (all residuals)" begin
            for name in probs_NCL # several tests
                nlp = CUTEstModel(name)
                test_name = name * " problem resolution"
                @testset "$test_name" begin
                    @test NCLSolve(nlp).solver_specific[:internal_msg] == Symbol("Solve_Succeeded")
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
