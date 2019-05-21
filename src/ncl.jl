# mymethod
# ** Important
# ! Alert, problem
# ? Question
# TODO 

using LinearAlgebra


"""
Creates the current subproblem (iteration k) for the NCL method.
Arguments :
    nlp : optimization problem considered
    y_k : current langrangian multiplicator
"""
function nc_k(nlp, y_k) # TODO: link with NLPModels
end

"""
Checks if the nlp initiql problem is solved with the x, y, z points withine the ω tolerance
"""

function NCO_solved(x, y, z, ω, nlp) # TODO
    return(true)
end


"""
NCL method implementation. See https://www.researchgate.net/publication/325480151_Stabilized_Optimization_Via_an_NCL_Algorithm for further explications on the method
Arguments : 
    - nlp : optimization problem described by the modelization NLPmodels.jl (voir https://github.com/JuliaSmoothOptimizers/NLPModels.jl)
      nlp is the generic problem you want to solve
Returns :

"""

function ncl(nlp, maxIt::Int64)
    # ** I. Parameters
    ρ_k = 1 # step
    τ = 10 # scale (used to update the ρ_k step)
    α = 0.5 # Constant (α needs to be < 1)
    β = 1 # Constant

    ω_end = 1 # global tolerance
    ω_k = 1 # sub problem tolerance
    η_end = 1 # global infeasability
    η_k = 2 # sub problem infeasability

    # ** II. Optimization loop
    k = 0
    converged = false
    while k < maxIt and !converged
        k += 1
        NC_k =  nc_k(nlp, y_k) # Sub problem modelisation
        x_k, y_k, r_k, z_k = KNITRO(NC_k, ω_k) # TODO: link with KNITRO/IPOPT
        if LinearAlgebra.norm(q,Inf) <= max(η_k, η_end)
            y_k = y_k + ρ_k * r_k
            if NCO_solved(x_k, y_k, z_k, ω_end, nlp)
                converged = true
            end
            η_k = η_k / (1 + ρ_k ^ β)
        else
            ρ_k = τ * ρ_k
            η_k = η_end / (1 + ρ_k ^ α) # ? η_end ou η_0, cf article
        end
    end

    return( x_k, y_k, r_k, z_k, converged) # converged tells us if the solution returned is optimal or not
end