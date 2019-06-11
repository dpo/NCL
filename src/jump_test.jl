using JuMP
using Ipopt

m = Model(with_optimizer(Ipopt.Optimizer))
@variable(m, x_1 >= 0)
@variable(m, x_2 >= 0)
@NLconstraint(m, (1-x_1)^3 - x_2 >= 0)
@NLobjective(m, Min, (x_1 - 2)^2 + x_2^2)

JuMP.optimize!(m)
