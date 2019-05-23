using NLPModels
using LinearAlgebra
include("ncl.jl")



f(x) = x[1]
x0 = [1]
lvar = [-1]
uvar = [12]
lcon = [0,0]
ucon = [Inf,Inf]
c(x) = [2*x[1], 3*x[1]]
nlp = ADNLPModel(f, x0; lvar=lvar, uvar=uvar, c=c, lcon=lcon, ucon=ucon)
println(nlp)
println(ncl(nlp, 2))