#include("../src/pb_set_resol.jl")

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
			pb_set_resolution_data(cutest_pb_set = ["HS1", "HS12"], 
														 nlp_pb_set = [nlp], 
														 ampl_pb_dir_path = "./test/",
														 ampl_pb_set = ["hs13"], 
														 solver = ["nclres", "nclkkt", "ipopt"], 
														 create_latex_table=true, 
														 latex_table_name = "test_latex.tex", 
														 create_profile = true, 
														 profile_name = "test_profile.svg"
														 )

			@test isfile("./res/test_latex.tex")
			isfile("./res/test_latex.tex") && rm("./res/test_latex.tex")
			@test isfile("./res/test_profile.svg")
			isfile("./res/test_profile.svg") && rm("./res/test_profile.svg")
    end
end