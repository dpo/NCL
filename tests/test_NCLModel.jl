using NLPModels
using Test

include("../src/NCLModel.jl")



"""
#################################
# Unitary tests for NLCModel.jl #
#################################
"""
function test_NLCModel(test::Bool) ::Test.DefaultTestSet
    # Test parameters
        ρ = 1.
        y = [2., 1.]
        g = Vector{Float64}(undef,4)
        cx = Vector{Float64}(undef,4)
        
        hrows = [1, 2, 2, 3, 4]
        hcols = [1, 1, 2, 3, 4]
        hvals = Vector{Float64}(undef,5)
        Hv = Vector{Float64}(undef,4)

        jrows = [1, 2, 3, 4, 1, 2, 3, 4, 3, 4]
        jcols = [1, 1, 1, 1, 2, 2, 2, 2, 3, 4]
        jvals = Vector{Float64}(undef,10)
        Jv = Vector{Float64}(undef,4)
        
    # Test problem
        f(x) = x[1] + x[2]
        x0 = [0.5, 0.5]
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
        nlp::ADNLPModel = ADNLPModel(f, x0 ; lvar=lvar, uvar=uvar, c=c, lcon=lcon, ucon=ucon, name=name, lin = [1,3])
        nlc_nlin_res::NCLModel = NCLModel(nlp ; res_lin_cons = false)

        nlc_nlin_res.y = y
        nlc_nlin_res.ρ = ρ


        nlc_cons_res::NCLModel = NCLModel(nlp, res_lin_cons = true)
        nlc_cons_res.ρ = ρ

    # Unitary tests
        if test
            @testset "NCLModel. No linear residuals" begin
                @testset "NCLModel struct" begin
                    @testset "NCLModel struct information about nlp" begin
                        @test nlc_nlin_res.nvar_x == 2 
                        @test nlc_nlin_res.nvar_r == 2 # two non linear constraint, so two residues
                        @test nlc_nlin_res.minimize == true
                        @test nlc_nlin_res.jres == [2,4]
                    end

                    @testset "NCLModel struct constant parameters" begin
                        @test nlc_nlin_res.nvar == 4 # 2 x, 2 r
                        @test nlc_nlin_res.meta.lvar == [0., 0., -Inf, -Inf] # no bounds for residues
                        @test nlc_nlin_res.meta.uvar == [1., 1., Inf, Inf]
                        @test nlc_nlin_res.meta.x0 == [0.5, 0.5, 1., 1.]
                        @test nlc_nlin_res.meta.y0 == [0., 0., 0., 0.]
                        @test nlc_nlin_res.y == y
                        @test length(nlc_nlin_res.y) == nlc_nlin_res.nvar_r
                        @test nlc_nlin_res.meta.nnzj == nlp.meta.nnzj + 2 # 2 residues, one for each non linear constraint
                        @test nlc_nlin_res.meta.nnzh == nlp.meta.nnzh + 2 # add a digonal of ρ
                    end
                end

                @testset "NCLModel f" begin
                    @test obj(nlc_nlin_res, [0., 0., 0., 0.]) == 0.
                    @test obj(nlc_nlin_res, [0.5, 0.5, 0., -1.]) == 1. - 1. + 0.5 * ρ * 1.
                end

                @testset "NCLModel ∇f" begin 
                    @testset "NCLModel grad()" begin
                        @test grad(nlc_nlin_res, [0., 0., 0., 0.]) == [1., 1., 2., 1.]
                        @test grad(nlc_nlin_res, [0.5, 0.5, 0., -1.]) == [1., 1., 2., 1. - ρ]
                    end

                    @testset "NCLModel grad!()" begin
                        @test grad!(nlc_nlin_res, [0., 0., 0., 0.], g) == [1., 1., 2., 1.]
                        @test grad!(nlc_nlin_res, [0.5, 0.5, 0., -1.], zeros(4)) == [1., 1., 2., 1. - ρ]
                    end
                end

                @testset "NCLModel hessian of the lagrangian" begin
                    @testset "NCLModel hessian of the lagrangian hess()" begin
                        @test hess(nlc_nlin_res, [0., 0., 0., 0.], y=zeros(Float64,4)) == [0. 0. 0. 0. ; 
                                                                                0. 0. 0. 0. ;
                                                                                0. 0. ρ  0. ;
                                                                                0. 0. 0. ρ]
                        @test hess(nlc_nlin_res, nlc_nlin_res.meta.x0, y=[1.,1.,1.,1.]) == [2. 0. 0. 0. ; #not symetrical because only the lower triangle is returned by hess
                                                                        1. 0. 0. 0. ;
                                                                        0. 0. ρ  0. ;
                                                                        0. 0. 0. ρ]
                    end               

                    @testset "NCLModel hessian of the lagrangian hess_coord()" begin
                        @test hess_coord(nlc_nlin_res, [0., 0., 0., 0.], y = zeros(Float64,4))[1][nlp.meta.nnzh+1 : nlp.meta.nnzh+2] == [3, 4]
                        @test hess_coord(nlc_nlin_res, nlc_nlin_res.meta.x0, y = [1.,1.,1.,1.])[1][nlp.meta.nnzh+1 : nlp.meta.nnzh+2] == [3, 4]

                        @test hess_coord(nlc_nlin_res, [0., 0., 0., 0.], y = zeros(Float64,4))[2][nlp.meta.nnzh+1 : nlp.meta.nnzh+2] == [3, 4]
                        @test hess_coord(nlc_nlin_res, nlc_nlin_res.meta.x0, y = [1.,1.,1.,1.])[2][nlp.meta.nnzh+1 : nlp.meta.nnzh+2] == [3, 4]

                        @test hess_coord(nlc_nlin_res, [0., 0., 0., 0.], y = zeros(Float64,4))[3][nlp.meta.nnzh+1 : nlp.meta.nnzh+2] == [ρ, ρ]
                        @test hess_coord(nlc_nlin_res, nlc_nlin_res.meta.x0, y = [1.,1.,1.,1.])[3][nlp.meta.nnzh+1 : nlp.meta.nnzh+2] == [ρ, ρ]
                    end

                    @testset "NCLModel hessian of the lagrangian hess_coord!()" begin
                        @test hess_coord!(nlc_nlin_res, [0., 0., 0., 0.], hrows, hcols, hvals, y = zeros(Float64,4))[1] == hrows
                        @test hess_coord!(nlc_nlin_res, nlc_nlin_res.meta.x0, hrows, hcols, hvals, y = [1.,1.,1.,1.])[1] == hrows

                        @test hess_coord!(nlc_nlin_res, [0., 0., 0., 0.], hrows, hcols, hvals, y = zeros(Float64,4))[2] == hcols
                        @test hess_coord!(nlc_nlin_res, nlc_nlin_res.meta.x0, hrows, hcols, hvals, y = [1.,1.,1.,1.])[2] == hcols

                        @test hess_coord!(nlc_nlin_res, [0., 0., 0., 0.], hrows, hcols, hvals, y = zeros(Float64,4))[3][nlp.meta.nnzh+1 : nlp.meta.nnzh+2] == [ρ, ρ]
                        @test hess_coord!(nlc_nlin_res, nlc_nlin_res.meta.x0, hrows, hcols, hvals, y = [1.,1.,1.,1.])[3][nlp.meta.nnzh+1 : nlp.meta.nnzh+2] == [ρ, ρ]
                    end


                    @testset "NCLModel hessian of the lagrangian hess_structure()" begin
                        @test hess_structure(nlc_nlin_res)[1] == vcat(hess_structure(nlc_nlin_res.nlp)[1], [3, 4])
                        @test hess_structure(nlc_nlin_res)[2] == vcat(hess_structure(nlc_nlin_res.nlp)[2], [3, 4])
                    end

                    @testset "NCLModel hessian of the lagrangian hprod()" begin
                        @test hprod(nlc_nlin_res, nlc_nlin_res.meta.x0, [1.,2.,3.,4.], y = [1.,1.,1.,1.]) == [4,1,3*ρ,4*ρ]
                    end

                    @testset "NCLModel hessian of the lagrangian hprod!()" begin
                        @test hprod!(nlc_nlin_res, nlc_nlin_res.meta.x0, [1.,2.,3.,4.], y = [1.,1.,1.,1.], Hv) == [4,1,3*ρ,4*ρ]
                    end
                end

                @testset "NCLModel constraint" begin
                    @testset "NCLModel constraint cons()" begin
                        @test size(cons(nlc_nlin_res, [1.,1.,0.,1.]), 1) == 4
                        @test cons(nlc_nlin_res, [1.,1.,0.,1.]) == [0.,2.,0.,2.]
                        @test cons(nlc_nlin_res, [1.,0.5,1.,1.]) == [0.5,2.5,0.5,1.5]
                    end
                    @testset "NCLModel constraint cons!()" begin
                        @test size(cons!(nlc_nlin_res, [1.,1.,0.,1.], cx), 1) == 4
                        @test cons!(nlc_nlin_res, [1.,1.,0.,1.], cx) == [0.,2.,0.,2.]
                        @test cons!(nlc_nlin_res, [1.,0.5,1.,1.], cx) == [0.5,2.5,0.5,1.5]
                    end
                end

                @testset "NCLModel constraint jacobian" begin
                    @testset "NCLModel constraint jac()" begin
                        @test jac(nlc_nlin_res, [1.,1.,0.,1.]) == [1 -1 0 0 ;
                                                        2  1 1 0 ;
                                                        1 -1 0 0 ;
                                                        1  1 0 1 ]

                        @test jac(nlc_nlin_res, [1.,0.5,1.,1.]) == [ 1 -1  0  0 ;
                                                            2  1  1  0 ;
                                                            1 -1  0  0 ;
                                                            0.5 1  0  1]
                    end
                    
                    @testset "NCLModel constraint jac_coord()" begin
                        @test jac_coord(nlc_nlin_res, [1.,1.,0.,1.])[1][9:10] == [2,4]
                        @test jac_coord(nlc_nlin_res, [1.,1.,0.,1.])[2][9:10] == [3,4]
                        @test jac_coord(nlc_nlin_res, [1.,0.5,1.,1.])[3][9:10] == [1,1]
                    end

                    @testset "NCLModel constraint jac_coord!()" begin
                        @test jac_coord!(nlc_nlin_res, [1.,1.,0.,1.], jrows, jcols, jvals)[1] == jrows
                        @test jac_coord!(nlc_nlin_res, [1.,1.,0.,1.], jrows, jcols, jvals)[2] == jcols
                        @test jac_coord!(nlc_nlin_res, [1.,1.,0.,1.], jrows, jcols, jvals)[3] == [1,2,1,1,-1,1,-1,1,1,1]
                        @test jac_coord!(nlc_nlin_res, [1.,0.5,1.,1.], jrows, jcols, jvals)[3] == [1,2,1,0.5,-1,1,-1,1,1,1]
                    end

                    @testset "NCLModel constraint jac_struct()" begin
                        @test jac_structure(nlc_nlin_res)[1][9:10] == [2,4]
                        @test jac_structure(nlc_nlin_res)[2][9:10] == [3,4]
                    end

                    @testset "NCLModel constraint jprod()" begin
                        @test jprod(nlc_nlin_res, [1.,1.,0.,1.], [1.,1.,1.,1.]) == [0,4,0,3]
                        @test jprod(nlc_nlin_res, [1.,0.5,1.,1.], [0.,1.,0.,1.]) == [-1,1,-1,2]
                    end

                    @testset "NCLModel constraint jprod!()" begin
                        @test jprod!(nlc_nlin_res, [1.,1.,0.,1.], [1.,1.,1.,1.], Jv) == [0,4,0,3]
                        @test jprod!(nlc_nlin_res, [1.,0.5,1.,1.], [0.,1.,0.,1.], Jv) == [-1,1,-1,2]
                    end

                    @testset "NCLModel constraint jtprod()" begin
                        @test jtprod(nlc_nlin_res, [1.,1.,0.,1.], [1.,1.,1.,1.]) == [5,0,1,1]
                        @test jtprod(nlc_nlin_res, [1.,0.5,1.,1.], [0.,1.,0.,1.]) == [2.5,2,1,1]
                    end

                    @testset "NCLModel constraint jtprod!()" begin
                        @test jtprod!(nlc_nlin_res, [1.,1.,0.,1.], [1.,1.,1.,1.], Jv) == [5,0,1,1]
                        @test jtprod!(nlc_nlin_res, [1.,0.5,1.,1.], [0.,1.,0.,1.], Jv) == [2.5,2,1,1]
                    end
                end
            end

            @testset "NCLModel. All residuals" begin
                @testset "NCLModel struct" begin
                    @testset "NCLModel struct information about nlp" begin
                        @test nlc_cons_res.nvar_x == 2 
                        @test nlc_cons_res.nvar_r == 4 # two non linear constraint, so two residues
                        @test nlc_cons_res.minimize == true
                        @test nlc_cons_res.jres == []
                    end

                    @testset "NCLModel struct constant parameters" begin
                        @test nlc_cons_res.nvar == 6 # 2 x, 4 r
                        @test nlc_cons_res.meta.lvar == [0., 0., -Inf, -Inf, -Inf, -Inf] # no bounds for residues
                        @test nlc_cons_res.meta.uvar == [1., 1., Inf, Inf, Inf, Inf]
                        @test nlc_cons_res.meta.x0 == [0.5, 0.5, 1., 1., 1., 1.]
                        @test nlc_cons_res.meta.y0 == [0., 0., 0., 0.]
                        @test nlc_cons_res.y == [1., 1., 1., 1.]
                        @test length(nlc_cons_res.y) == nlc_cons_res.nvar_r
                        @test nlc_cons_res.meta.nnzj == nlp.meta.nnzj + 4 # 2 residues, one for each constraint
                        @test nlc_cons_res.meta.nnzh == nlp.meta.nnzh + 4 # add a digonal of ρ
                    end
                end

                @testset "NCLModel f" begin
                    @test obj(nlc_cons_res, [0., 0., 0., 0., 0., 0.]) == 0.
                    @test obj(nlc_cons_res, [0.5, 0.5, 0., -1., 0., 1.]) == 1. + 0. + 0.5 * ρ * (1. + 1.)
                end

                @testset "NCLModel ∇f" begin 
                    @testset "NCLModel grad()" begin
                        @test grad(nlc_cons_res, [0., 0., 0., 0., 0., 0.]) == [1., 1., 1., 1., 1., 1.]
                        @test grad(nlc_cons_res, [0.5, 0.5, 0., -1., 0., 1.]) == [1., 1., 1., 1. - ρ, 1., 1 + ρ]
                    end

                    @testset "NCLModel grad!()" begin
                        @test grad!(nlc_cons_res, [0., 0., 0., 0., 0., 0.], vcat(g, [1,2])) == [1., 1., 1., 1., 1., 1.]
                        @test grad!(nlc_cons_res, [0.5, 0.5, 0., -1., 0., 1.], zeros(6)) == [1., 1., 1., 1. - ρ, 1., 1 + ρ]
                    end
                end

                @testset "NCLModel hessian of the lagrangian" begin
                    @testset "NCLModel hessian of the lagrangian hess()" begin
                        @test hess(nlc_cons_res, [0., 0., 0., 0.], y=zeros(Float64,6)) == [0. 0. 0. 0. 0. 0. ; 
                                                                                        0. 0. 0. 0. 0. 0. ;
                                                                                        0. 0. ρ  0. 0. 0. ;
                                                                                        0. 0. 0. ρ  0. 0. ;
                                                                                        0. 0. 0. 0. ρ  0. ;
                                                                                        0. 0. 0. 0. 0. ρ ]
                        @test hess(nlc_cons_res, nlc_cons_res.meta.x0, y=[1.,1.,1.,1.]) == [2. 0. 0. 0. 0. 0. ; #not symetrical because only the lower triangle is returned by hess
                                                                                            1. 0. 0. 0. 0. 0. ;
                                                                                            0. 0. ρ  0. 0. 0. ;
                                                                                            0. 0. 0. ρ  0. 0. ;
                                                                                            0. 0. 0. 0. ρ  0. ;
                                                                                            0. 0. 0. 0. 0. ρ ]
                    end               

                    @testset "NCLModel hessian of the lagrangian hess_coord()" begin
                        @test hess_coord(nlc_cons_res, [0., 0., 0., 0., 0., 0.], y = zeros(Float64,6))[1][nlp.meta.nnzh+1 : nlp.meta.nnzh+nlc_cons_res.nvar_r] == [3, 4, 5, 6]
                        @test hess_coord(nlc_cons_res, nlc_cons_res.meta.x0, y = [1.,1.,1.,1.,1.,1.])[1][nlp.meta.nnzh+1 : nlp.meta.nnzh+nlc_cons_res.nvar_r] == [3, 4, 5, 6]

                        @test hess_coord(nlc_cons_res, [0., 0., 0., 0., 0., 0.], y = zeros(Float64,6))[2][nlp.meta.nnzh+1 : nlp.meta.nnzh+nlc_cons_res.nvar_r] == [3, 4, 5, 6]
                        @test hess_coord(nlc_cons_res, nlc_cons_res.meta.x0, y = [1.,1.,1.,1.,1.,1.])[2][nlp.meta.nnzh+1 : nlp.meta.nnzh+nlc_cons_res.nvar_r] == [3, 4, 5, 6]

                        @test hess_coord(nlc_cons_res, [0., 0., 0., 0., 0., 0.], y = zeros(Float64,6))[3][nlp.meta.nnzh+1 : nlp.meta.nnzh+nlc_cons_res.nvar_r] == [ρ, ρ, ρ, ρ]
                        @test hess_coord(nlc_cons_res, nlc_cons_res.meta.x0, y = [1.,1.,1.,1.,1.,1.])[3][nlp.meta.nnzh+1 : nlp.meta.nnzh+nlc_cons_res.nvar_r] == [ρ, ρ, ρ, ρ]
                    end

                    @testset "NCLModel hessian of the lagrangian hess_coord!()" begin
                        @test hess_coord!(nlc_cons_res, [0., 0., 0., 0., 0., 0.], vcat(hrows, [5, 6]), vcat(hcols, [5, 6]), vcat(hvals, [5, 6]), y = zeros(Float64,6))[1] == vcat(hrows, [5, 6])
                        @test hess_coord!(nlc_cons_res, nlc_cons_res.meta.x0, vcat(hrows, [5, 6]), vcat(hcols, [5, 6]), vcat(hvals, [5, 6]), y = [1.,1.,1.,1.,1.,1.])[1] == vcat(hrows, [5, 6])

                        @test hess_coord!(nlc_cons_res, [0., 0., 0., 0., 0., 0.], vcat(hrows, [5, 6]), vcat(hcols, [5, 6]), vcat(hvals, [5, 6]), y = zeros(Float64,6))[2] == vcat(hcols, [5, 6])
                        @test hess_coord!(nlc_cons_res, nlc_cons_res.meta.x0, vcat(hrows, [5, 6]), vcat(hcols, [5, 6]), vcat(hvals, [5, 6]), y = [1.,1.,1.,1.,1.,1.])[2] == vcat(hcols, [5, 6])

                        @test hess_coord!(nlc_cons_res, [0., 0., 0., 0., 0., 0.], vcat(hrows, [5, 6]), vcat(hcols, [5, 6]), vcat(hvals, [5, 6]), y = zeros(Float64,6))[3][nlp.meta.nnzh+1 : nlp.meta.nnzh+nlc_cons_res.nvar_r] == [ρ, ρ, ρ, ρ]
                        @test hess_coord!(nlc_cons_res, nlc_cons_res.meta.x0, vcat(hrows, [5, 6]), vcat(hcols, [5, 6]), vcat(hvals, [5, 6]), y = [1.,1.,1.,1.,1.,1.])[3][nlp.meta.nnzh+1 : nlp.meta.nnzh+nlc_cons_res.nvar_r] == [ρ, ρ, ρ, ρ]
                    end


                    @testset "NCLModel hessian of the lagrangian hess_structure()" begin
                        @test hess_structure(nlc_cons_res)[1] == vcat(hess_structure(nlc_cons_res.nlp)[1], [3, 4, 5, 6])
                        @test hess_structure(nlc_cons_res)[2] == vcat(hess_structure(nlc_cons_res.nlp)[2], [3, 4, 5, 6])
                    end

                    @testset "NCLModel hessian of the lagrangian hprod()" begin
                        @test hprod(nlc_cons_res, nlc_cons_res.meta.x0, [1.,2.,3.,4.,5.,6.], y = [1.,1.,1.,1.,1.,1.]) == [4,1,3*ρ,4*ρ,5*ρ,6*ρ]
                    end

                    @testset "NCLModel hessian of the lagrangian hprod!()" begin
                        @test hprod!(nlc_cons_res, nlc_cons_res.meta.x0, [1.,2.,3.,4.,5.,6.], y = [1.,1.,1.,1.,1.,1.], vcat(Hv, [0.,0.])) == [4,1,3*ρ,4*ρ,5*ρ,6*ρ]
                    end
                end

                @testset "NCLModel constraint" begin
                    @testset "NCLModel constraint cons()" begin
                        @test size(cons(nlc_cons_res, [1.,1.,0.,1.,1.,1.]), 1) == 4
                        @test cons(nlc_cons_res, [1.,1.,0.,1.,1.,1.]) == [0.,3.,1.,2.]
                        @test cons(nlc_cons_res, [1.,0.5,1.,1.,0.,-1.]) == [1.5,2.5,0.5,-0.5]
                    end
                    @testset "NCLModel constraint cons!()" begin
                        @test size(cons!(nlc_cons_res, [1.,1.,0.,1.,1.,1.], cx), 1) == 4
                        @test cons!(nlc_cons_res, [1.,1.,0.,1.,1.,1.], cx) == [0.,3.,1.,2.]
                        @test cons!(nlc_cons_res, [1.,0.5,1.,1.,0.,-1.], cx) == [1.5,2.5,0.5,-0.5]
                    end
                end

                @testset "NCLModel constraint jacobian" begin
                    @testset "NCLModel constraint jac()" begin
                        @test jac(nlc_cons_res, [1.,1.,0.,1.,1.,1.]) == [1 -1  1  0  0  0;
                                                                        2  1  0  1  0  0;
                                                                        1 -1  0  0  1  0;
                                                                        1  1  0  0  0  1]

                        @test jac(nlc_cons_res, [1.,0.5,1.,1.,0.,-1.]) == [1  -1  1  0  0  0;
                                                                        2   1  0  1  0  0;
                                                                        1  -1  0  0  1  0;
                                                                        0.5  1  0  0  0  1]
                    end
                    
                    @testset "NCLModel constraint jac_coord()" begin
                        @test jac_coord(nlc_cons_res, [1.,1.,0.,1.,1.,1.])[1][9:12] == [1,2,3,4]
                        @test jac_coord(nlc_cons_res, [1.,1.,0.,1.,1.,1.])[2][9:12] == [3,4,5,6]
                        @test jac_coord(nlc_cons_res, [1.,0.5,1.,1.,0.,-1.])[3][9:12] == [1,1,1,1]
                    end

                    @testset "NCLModel constraint jac_coord!()" begin
                        @test jac_coord!(nlc_cons_res, [1.,1.,0.,1.,1.,1.], vcat(jrows, [1,2]), vcat(jcols, [0,0]), vcat(jvals, [1,21]))[1] == vcat(jrows, [1,2])
                        @test jac_coord!(nlc_cons_res, [1.,1.,0.,1.,1.,1.], vcat(jrows, [1,2]), vcat(jcols, [0,0]), vcat(jvals, [1,21]))[2] == vcat(jcols, [0,0])
                        @test jac_coord!(nlc_cons_res, [1.,1.,0.,1.,1.,1.], vcat(jrows, [1,2]), vcat(jcols, [0,0]), vcat(jvals, [1,21]))[3] == [1,2,1,1,-1,1,-1,1,1,1,1,1]
                        @test jac_coord!(nlc_cons_res, [1.,0.5,1.,1.,0.,-1.], vcat(jrows, [1,2]), vcat(jcols, [0,0]), vcat(jvals, [1,21]))[3] == [1,2,1,0.5,-1,1,-1,1,1,1,1,1]
                    end

                    @testset "NCLModel constraint jac_struct()" begin
                        @test jac_structure(nlc_cons_res)[1][9:12] == [1,2,3,4]
                        @test jac_structure(nlc_cons_res)[2][9:12] == [3,4,5,6]
                    end

                    @testset "NCLModel constraint jprod()" begin
                        @test jprod(nlc_cons_res, [1.,1.,0.,1.,1.,1.], [1.,1.,1.,1.,1.,1.]) == [1,4,1,3]
                        @test jprod(nlc_cons_res, [1.,0.5,1.,1.,0.,-1.], [0.,1.,0.,1.,-1.,-1.]) == [-1,2,-2,0]
                    end

                    @testset "NCLModel constraint jprod!()" begin
                        @test jprod!(nlc_cons_res, [1.,1.,0.,1.,1.,1.], [1.,1.,1.,1.,1.,1.], Jv) == [1,4,1,3]
                        @test jprod!(nlc_cons_res, [1.,0.5,1.,1.,0.,-1.], [0.,1.,0.,1.,-1.,-1.], Jv) == [-1,2,-2,0]
                    end

                    @testset "NCLModel constraint jtprod()" begin
                        @test jtprod(nlc_cons_res, [1.,1.,0.,1.,1.,1.], [1.,1.,1.,1.]) == [5,0,1,1,1,1]
                        @test jtprod(nlc_cons_res, [1.,0.5,1.,1.,0.,-1.], [0.,1.,0.,1.]) == [2.5,2,0,1,0,1]
                    end

                    @testset "NCLModel constraint jtprod!()" begin
                        @test jtprod!(nlc_cons_res, [1.,1.,0.,1.,1.,1.], [1.,1.,1.,1.], vcat(Jv, [0,1])) == [5,0,1,1,1,1]
                        @test jtprod!(nlc_cons_res, [1.,0.5,1.,1.,0.,-1.], [0.,1.,0.,1.], vcat(Jv, [0,1])) == [2.5,2,0,1,0,1]
                    end
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