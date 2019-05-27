using NLPModels
using LinearAlgebra
using Test

# TODO : Faire les tests unitaires
#include("ncl.jl")
include("NLCModel.jl")

# Paramètres de test
ρ = 1.
y = [2., 1.]


f(x) = x[1] + x[2]
x0 = [1., 1.]
lvar = [0., 0.]
uvar = [1., 1.]
lcon = [0.,0.]
ucon = [Inf,Inf]
c(x) = [x[1]+0.5-x[2], x[2]-(x[1]-0.5)]
nlp = ADNLPModel(f, x0; lvar=lvar, uvar=uvar, c=c, lcon=lcon, ucon=ucon)
nlc = NLCModel(nlp, y, ρ, false)::NLCModel
g = [0., 4., 1]


@testset "NLCModel" begin
    @testset "NLCModel f" begin
        @test obj(nlc, [0., 0., 0., 0.]) == 0.
        @test obj(nlc, nlc.meta.x0) == (1. + 1.) + 3. + 0.5 * ρ * 2
    end

    @testset "NLCModel ∇f" begin 
        @testset "NLCModel grad()" begin
            @test grad(nlc, [0., 0., 0., 0.]) == [1., 1., 2., 1.]
            @test grad(nlc, nlc.meta.x0) == [1., 1., 3., 2.]
        end

        @testset "NLCModel grad!()" begin
            @test grad!(nlc, [0., 0., 0., 0.], g) == [1., 1., 2., 1.]
            @test grad!(nlc, nlc.meta.x0, zeros(4)) == [1., 1., 3., 2.]
        end
    end

    @testset "NLCModel hessian of the lagrangian" begin
        @test grad(nlc, [0., 0., 0., 0.]) == [1., 1., 2., 1.]
        @test grad(nlc, nlc.meta.x0) == [1., 1., 3., 2.]
    end
    
end