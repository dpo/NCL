#include("../src/pb_set_resol.jl")
using NCL
using Test
using NLPModels
"""
##############################
# Unit tests for NCLSolve.jl #
##############################
"""
function test_pb_set_resol(test::Bool) ::Test.DefaultTestSet
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

    @testset "pb_set_resol" begin
        pb_set_resolution_data(cutest_pb_set = ["HS1", "HS10"], nlp_pb_set = [nlp], solver = ["nclres", "ipopt"])
        @test isfile("./res/default_latex_table_name.tex")
        rm("./res/default_latex_table_name.tex")
        @test isfile("./res/default_profile_name.svg")
        rm("./res/default_profile_name.svg")
    end
end