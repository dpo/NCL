
using NLPModels

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

name_nlp = "Unitary test problem"
nlp = ADNLPModel(f, x0; lvar=lvar, uvar=uvar, c=c, lcon=lcon, ucon=ucon, name=name_nlp)::ADNLPModel


typeof(jac(nlp, nlp.meta.x0))
typeof(hess(nlp, nlp.meta.x0))