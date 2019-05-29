# comment
# ** Important
# ! Alert, problem
# ? Question
# TODO 

using LinearAlgebra
using NLPModels
using SolverTools
using Ipopt

#include("../src/NLPModelsIpopt_perso.jl")
#import ipopt
#using NLPModelsKnitro #! not found...
using NLPModelsIpopt
include("NLCModel.jl")



# ? Mais appel a la fonction long aussi. Inline faisable avec @inline si interessant




"""
Tests if the (x, y) is a solution of the KKT conditions of the nlp problem (nlp follows the NLPModels.jl formalism) within ω as a tolerance
Note: the lagrangian is considered as :
    l(x, y) = f(x) - y' * c(x)          (!!! -, not + !!!)
"""
function NLPModel_solved(nlp::AbstractNLPModel, x::Vector{<:Real}, y::Vector{<:Real}, z_U::Vector{<:Real}, z_L::Vector{<:Real}, ω::Real, printing::Bool) ::Bool
    # Bounds constraints
        for i in 1:nlp.meta.nvar 
            if !(nlp.meta.lvar[i] - ω <= x[i] <= nlp.meta.uvar[i] + ω) 
                if printing
                    println("variable " * string(i) * " out of bounds + tolerance") 
                end
                return false
            end

            # Complementarity for bounds
            if nlp.meta.lvar[i] > -Inf # ? Optimiser avec jupp, jlow ?
                if z_L[i] * (x[i] - nlp.meta.lvar[i]) > ω  
                    if printing
                        println("complementarity not respected, see lowerbound of var" * string(i))
                        @show z_L[i] * (nlp.meta.lvar[i] - x[i])
                    end
                    return false
                end
            end
            if nlp.meta.uvar[i] < Inf # ? Optimiser avec jupp, jlow ?
                if z_U[i] * (nlp.meta.uvar[i] - x[i]) > ω  
                    if printing
                        println("complementarity not respected, see upperbound of var" * string(i))
                        @show z_U[i] * (nlp.meta.uvar[i] - x[i])
                    end
                    return false
                end
            end
        end
    println("Faisable")
    # Other constraints
        c_x = cons(nlp, x)
        
        for i in 1:nlp.meta.ncon
            if !(nlp.meta.lcon[i] - ω <= c_x[i] <= nlp.meta.ucon[i] + ω)
                if printing
                    println("constraint " * string(i) * " out of bounds + tolerance") 
                end
                return false 
            end

            if i in nlp.meta.jinf
                error("Infeasable problem passed to NLPModel_solved.\n
                       Check the constraint" * string(i))
            end

            # Complementarity
            if nlp.meta.lcon[i] > -Inf # ? Optimiser avec jupp, jlow ?
                if abs(y[i] * (c_x[i] - nlp.meta.lcon[i])) > ω
                    if printing
                        println("complementarity not respected, see lower cons" * string(i))
                        @show y[i] * (c_x[i] - nlp.meta.lcon[i])
                    end
                    return false
                end
            end
            if nlp.meta.ucon[i] < Inf # ? Optimiser avec jupp, jlow ?
                if abs(y[i] * (nlp.meta.ucon[i] - c_x[i])) > ω  
                    if printing
                        println("complementarity not respected, see upper cons" * string(i))
                        @show y[i] * (nlp.meta.ucon[i] - c_x[i])
                    end
                    return false
                end
            end
        end
    println("realisable")
    # Lagrangian
        ∇f_x = grad(nlp, x)
        ∇lag = ∇f_x - jtprod(nlp, x, y) + z_L - z_U
        
        if norm(∇lag, Inf) > ω # Not a stationnary point for the lagrangian
            if printing
                @show jtprod(nlp, x, y)
                @show ∇lag
            end
            return false
        end

    return true # all the tests were passed, x, y respects feasability, complementarity not respected, see and ∇lag(x, y) almost = 0
end


function KNITRO(nlp, ω) # Juste pour pouvoir debugger petit à petit, sans avoir KNITRO
    return [1,2],[2,1],[2,1],[1,2,2,1]
end


"""
NCL method implementation. See https://www.researchgate.net/publication/325480151_Stabilized_Optimization_Via_an_NCL_Algorithm for further explications on the method
Arguments: 
    - nlp: optimization problem described by the modelization NLPModels.jl (voir https://github.com/JuliaSmoothOptimizers/NLPModels.jl)
      nlp is the generic problem you want to solve
Returns:
    x: position of the optimum found
    y: optimal lagrangian multiplicator
    r: value of constraints (around 0 if converged)
    z: lagrangian multiplicator for bounds constraints
    converged: a booleam, telling us if the progra; converged or reached the maximum of iterations fixed
"""
function ncl(nlc::NLCModel, maxIt::Int64, ipopt::Bool)
    # ** I. Names and variables
        Type = typeof(nlc.meta.x0[1])
        nlc.ρ = 1 # step
        τ = 10 # scale (used to update the ρ_k step)
        α = 0.5 # Constant (α needs to be < 1)
        β = 1 # Constant

        ω_end = 1 # global tolerance
        ω_k = 1 # sub problem tolerance
        η_end = 1 # global infeasability
        η_k = 2 # sub problem infeasability

        # initial points
        x_k = zeros(Type, nlc.nvar_x)
        r_k = zeros(Type, nlc.nvar_r)
        nlc.y = zeros(Type, nlc.meta.ncon)
        z_k = zeros(Type, length(nlc.meta.lvar) + length(nlc.meta.uvar))


    # ** II. Optimization loop
        k = 0
        converged = false

        while k < maxIt && !converged
            k += 1
            # ** II.1 Get subproblem's solution
                if ipopt
                    resolution_k = ipopt(nlc)::GenericExecutionStats
                    # Get variables
                    x_k = resolution_k.solution[1:nlc.nvar_x]
                    r_k = resolution_k.solution[nlc.nvar_x+1:nlc.nvar_r]

                    # Get multipliers
                    λ_k = resolution_k.solver_specific[:multipliers_con] 
                    z_k_U = resolution_k.solver_specific[:multipliers_U]
                    z_k_L = resolution_k.solver_specific[:multipliers_L]

                else
                    resolution_k = _knitro(nlc)::GenericExecutionStats
                    # Get variables
                    x_k = resolution_k.solution.x[1:nlc.nvar_x]
                    r_k = resolution_k.solution.x[nlc.nvar_x+1:nlc.nvar_r]

                    # Get multipliers
                    λ_k = resolution_k.solver_specific[:multipliers_con]
                    z_k_U = resolution_k.solver_specific[:multipliers_U] #! =[] dans ce cas, pas séparé par KNITRO...
                    z_k_L = resolution_k.solver_specific[:multipliers_L]
                end
                
                # TODO (recherche) : Points intérieurs à chaud...
                # TODO (recherche) : tester la proximité des multiplicateurs de renvoyés par KNITRO et le y_k du problème (si r petit, probablement proches.)

            # ** II.2 Treatment & update
                if norm(r_k,Inf) <= max(η_k, η_end) # The residue has decreased enough
                    
                    nlc.y = nlc.y + nlc.ρ * r_k # Updating multiplicator
                    η_k = η_k / (1 + nlc.ρ ^ β) # (heuristic)
                    
                    # ** II.2.1 Solution found ?
                    #tolerance
                    if norm(r_k,Inf) <= min(η_k, η_end)
                        converged = NLPModel_solved(nlc, x_k, λ_k) # TODO (~recherche) : Voir si nécessaire ou si lorsque la tolérance de KNITRO renvoyée est assez faible et r assez petit, on a aussi résolu le problème initial    
                    end

                else # The residue is to still too large
                    nlc.ρ = τ * nlc.ρ # increase the step # TODO (recherche) : Mieux choisir le pas pour avoir une meilleure convergence
                    η_k = η_end / (1 + nlc.ρ ^ α) # Change infeasability (heuristic) # ? (simple) η_end ou η_0, cf article
                    # TODO (recherche...) : update ω_k 
                end
                # ? Chez Nocedal & Wright, p.521, on a : ω_k = 1/nlc.ρ, nlc.ρ = 100ρ_k, η_k = 1/nlc.ρ^0.1
        end
    
    return x_k, nlc.y, r_k, z_k, converged # converged tells us if the solution returned is optimal or not
end