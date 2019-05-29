using NLPModels
using Test


include("../src/NLCModel.jl")

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
                -0.5,
                0.5]
        c(x) = [x[1] - x[2], # linear
                x[1]^2 + x[2], # non linear one, range constraint
                x[1] - x[2], # linear, lower bounded 
                x[1] * x[2]] # equality one

        name = "Unitary test problem"
        nlp = ADNLPModel(f, x0; lvar=lvar, uvar=uvar, c=c, lcon=lcon, ucon=ucon, name=name)::ADNLPModel
        nlc = NLCModel(nlp, y, ρ)::NLCModel

        println(cons(nlp, [1,1]))

    # Unitary tests
        if test
            @testset "NLCModel" begin
                @testset "NLCModel struct" begin
                    @testset "NLCModel struct information about nlp" begin
                        @test nlc.nvar_x == 2 
                        @test nlc.nvar_r == 2 # two non linear constraint, so two residues
                        @test nlc.minimize == true
                        @test nlc.jres == [2,4]
                    end

                    @testset "NLCModel struct constant parameters" begin
                        @test nlc.nvar == 4 # 2 x, 2 r
                        @test nlc.meta.lvar == [0., 0., -Inf, -Inf] # no bounds for residues
                        @test nlc.meta.uvar == [1., 1., Inf, Inf]
                        @test nlc.meta.x0 == [0.5, 0.5, 1., 1.]
                        @test nlc.meta.nnzj == nlp.meta.nnzj + 2 # 2 residues, one for each non linear constraint
                        @test nlc.meta.nnzh == nlp.meta.nnzh + 2 # add a digonal of ρ
                    end
                end

                @testset "NLCModel f" begin
                    @test obj(nlc, [0., 0., 0., 0.]) == 0.
                    @test obj(nlc, [0.5, 0.5, 0., -1.]) == 1. - 1. + 0.5 * ρ * 1.
                end

                @testset "NLCModel ∇f" begin 
                    @testset "NLCModel grad()" begin
                        @test grad(nlc, [0., 0., 0., 0.]) == [1., 1., 2., 1.]
                        @test grad(nlc, [0.5, 0.5, 0., -1.]) == [1., 1., 2., 1. - ρ]
                    end

                    @testset "NLCModel grad!()" begin
                        @test grad!(nlc, [0., 0., 0., 0.], g) == [1., 1., 2., 1.]
                        @test grad!(nlc, [0.5, 0.5, 0., -1.], zeros(4)) == [1., 1., 2., 1. - ρ]
                    end
                end

                @testset "NLCModel hessian of the lagrangian" begin
                    @testset "NLCModel hessian of the lagrangian hess()" begin
                        @test hess(nlc, [0., 0., 0., 0.], y=zeros(Float64,4)) == [0. 0. 0. 0. ; 
                                                                                0. 0. 0. 0. ;
                                                                                0. 0. ρ  0. ;
                                                                                0. 0. 0. ρ]
                        @test hess(nlc, nlc.meta.x0, y=[1.,1.,1.,1.]) == [2. 0. 0. 0. ; #not symetrical because only the lower triangle is returned by hess
                                                                        1. 0. 0. 0. ;
                                                                        0. 0. ρ  0. ;
                                                                        0. 0. 0. ρ]
                    end               

                    @testset "NLCModel hessian of the lagrangian hess_coord()" begin
                        @test hess_coord(nlc, [0., 0., 0., 0.], y = zeros(Float64,4))[1][nlp.meta.nnzh+1 : nlp.meta.nnzh+2] == [3, 4]
                        @test hess_coord(nlc, nlc.meta.x0, y = [1.,1.,1.,1.])[1][nlp.meta.nnzh+1 : nlp.meta.nnzh+2] == [3, 4]

                        @test hess_coord(nlc, [0., 0., 0., 0.], y = zeros(Float64,4))[2][nlp.meta.nnzh+1 : nlp.meta.nnzh+2] == [3, 4]
                        @test hess_coord(nlc, nlc.meta.x0, y = [1.,1.,1.,1.])[2][nlp.meta.nnzh+1 : nlp.meta.nnzh+2] == [3, 4]

                        @test hess_coord(nlc, [0., 0., 0., 0.], y = zeros(Float64,4))[3][nlp.meta.nnzh+1 : nlp.meta.nnzh+2] == [ρ, ρ]
                        @test hess_coord(nlc, nlc.meta.x0, y = [1.,1.,1.,1.])[3][nlp.meta.nnzh+1 : nlp.meta.nnzh+2] == [ρ, ρ]
                    end

                    @testset "NLCModel hessian of the lagrangian hess_coord!()" begin
                        @test hess_coord!(nlc, [0., 0., 0., 0.], hrows, hcols, hvals, y = zeros(Float64,4))[1] == hrows
                        @test hess_coord!(nlc, nlc.meta.x0, hrows, hcols, hvals, y = [1.,1.,1.,1.])[1] == hrows

                        @test hess_coord!(nlc, [0., 0., 0., 0.], hrows, hcols, hvals, y = zeros(Float64,4))[2] == hcols
                        @test hess_coord!(nlc, nlc.meta.x0, hrows, hcols, hvals, y = [1.,1.,1.,1.])[2] == hcols

                        @test hess_coord!(nlc, [0., 0., 0., 0.], hrows, hcols, hvals, y = zeros(Float64,4))[3][nlp.meta.nnzh+1 : nlp.meta.nnzh+2] == [ρ, ρ]
                        @test hess_coord!(nlc, nlc.meta.x0, hrows, hcols, hvals, y = [1.,1.,1.,1.])[3][nlp.meta.nnzh+1 : nlp.meta.nnzh+2] == [ρ, ρ]
                    end


                    @testset "NLCModel hessian of the lagrangian hess_structure()" begin
                        @test hess_structure(nlc)[1] == vcat(hess_structure(nlc.nlp)[1], [3, 4])
                        @test hess_structure(nlc)[2] == vcat(hess_structure(nlc.nlp)[2], [3, 4])
                    end

                    @testset "NLCModel hessian of the lagrangian hprod()" begin
                        @test hprod(nlc, nlc.meta.x0, [1,2,3,4], y = [1.,1.,1.,1.]) == [4,1,3*ρ,4*ρ]
                    end

                    @testset "NLCModel hessian of the lagrangian hprod!()" begin
                        @test hprod!(nlc, nlc.meta.x0, [1,2,3,4], y = [1.,1.,1.,1.], Hv) == [4,1,3*ρ,4*ρ]
                    end
                end

                @testset "NLCModel constraint" begin
                    @testset "NLCModel constraint cons()" begin
                        @test size(cons(nlc, [1.,1.,0.,1.]), 1) == 4
                        @test cons(nlc, [1.,1.,0.,1.]) == [0.,2.,0.,2.]
                        @test cons(nlc, [1.,0.5,1.,1.]) == [0.5,2.5,0.5,1.5]
                    end
                    @testset "NLCModel constraint cons!()" begin
                        @test size(cons!(nlc, [1.,1.,0.,1.], cx), 1) == 4
                        @test cons!(nlc, [1.,1.,0.,1.], cx) == [0.,2.,0.,2.]
                        @test cons!(nlc, [1.,0.5,1.,1.], cx) == [0.5,2.5,0.5,1.5]
                    end

                    @testset "NLCModel constraint jac()" begin
                        @test jac(nlc, [1.,1.,0.,1.]) == [1 -1 0 0 ;
                                                        2  1 1 0 ;
                                                        1 -1 0 0 ;
                                                        1  1 0 1 ]

                        @test jac(nlc, [1.,0.5,1.,1.]) == [ 1 -1  0  0 ;
                                                            2  1  1  0 ;
                                                            1 -1  0  0 ;
                                                            0.5 1  0  1]
                    end
                    
                    @testset "NLCModel constraint jac_coord()" begin
                        @test jac_coord(nlc, [1.,1.,0.,1.])[1][9:10] == [2,4]
                        @test jac_coord(nlc, [1.,1.,0.,1.])[2][9:10] == [3,4]
                        @test jac_coord(nlc, [1.,0.5,1.,1.])[3][9:10] == [1,1]
                    end

                    @testset "NLCModel constraint jac_coord!()" begin
                        @test jac_coord!(nlc, [1.,1.,0.,1.], jrows, jcols, jvals)[1] == jrows
                        @test jac_coord!(nlc, [1.,1.,0.,1.], jrows, jcols, jvals)[2] == jcols
                        @test jac_coord!(nlc, [1.,1.,0.,1.], jrows, jcols, jvals)[3] == [1,2,1,1,-1,1,-1,1,1,1]
                        @test jac_coord!(nlc, [1.,0.5,1.,1.], jrows, jcols, jvals)[3] == [1,2,1,0.5,-1,1,-1,1,1,1]
                    end

                    @testset "NLCModel constraint jac_struct()" begin
                        @test jac_structure(nlc)[1][9:10] == [2,4]
                        @test jac_structure(nlc)[2][9:10] == [3,4]
                    end

                    @testset "NLCModel constraint jprod()" begin
                        @test jprod(nlc, [1.,1.,0.,1.], [1.,1.,1.,1.]) == [0,4,0,3]
                        @test jprod(nlc, [1.,0.5,1.,1.], [0.,1.,0.,1.]) == [-1,1,-1,2]
                    end

                    @testset "NLCModel constraint jprod!()" begin
                        @test jprod!(nlc, [1.,1.,0.,1.], [1.,1.,1.,1.], Jv) == [0,4,0,3]
                        @test jprod!(nlc, [1.,0.5,1.,1.], [0.,1.,0.,1.], Jv) == [-1,1,-1,2]
                    end

                    @testset "NLCModel constraint jtprod()" begin
                        @test jtprod(nlc, [1.,1.,0.,1.], [1.,1.,1.,1.]) == [5,0,1,1]
                        @test jtprod(nlc, [1.,0.5,1.,1.], [0.,1.,0.,1.]) == [2.5,2,1,1]
                    end

                    @testset "NLCModel constraint jtprod!()" begin
                        @test jtprod!(nlc, [1.,1.,0.,1.], [1.,1.,1.,1.], Jv) == [5,0,1,1]
                        @test jtprod!(nlc, [1.,0.5,1.,1.], [0.,1.,0.,1.], Jv) == [2.5,2,1,1]
                    end
                end
            end
        end
end