# comment
# ** Important
# ! Alert, problem
# ? Question
# TODO 

using LinearAlgebra
using NLPModels
using SolverTools
using Ipopt
using NLPModelsIpopt

include("NLCModel.jl")



# ? Mais appel a la fonction long aussi. Inline faisable avec @inline si interessant




"""
Tests if the (x, y) is a solution of the KKT conditions of the nlp problem (nlp follows the NLPModels.jl formalism) within ω as a tolerance
Note: the lagrangian is considered as :
    l(x, y) = f(x) - y' * c(x)          (!!! -, not + !!!)
"""
function NLPModel_solved(nlp::AbstractNLPModel, x::Vector{<:Real}, y::Vector{<:Real}, ω::Real) ::Bool
    bounds = true
    feasable = true # by default, x is a feasable point. Then we check with the constraints
    complementarity = true
    optimal = true # same, we will check with KKT conditions

    for i in 1:nlp.meta.nvar 
        if !(nlp.meta.lvar[i] <= x[i] <= nlp.meta.uvar[i]) # bounds constraints
            return false
        end
    end

    c_x = cons(nlp, x)
    for i in 1:nlp.ncon
        if !(nlp.meta.lcon[i] <= c_x[i] <= nlp.meta.ucon[i]) # other constraints
            return false 
        end

        if i in nlp.jinf
            println("ERROR: Infeasable problem passed to NLPModel_solved. 
                    \n  Check the constraint" * string(i) * 
                    "\nFalse returned. ")
            return false
        end

        if y[i] * c_x[i] > ω # complementarity not respected
            return false 
        end
    end


    ∇lag = ∇f_x - jtprod(nlp, x, y)
    ∇f_x = grad(nlp, x)
    if norm(∇lag, Inf) > ω
        return false
    end

    return true # all the tests were passed, x, y respects feasability, complementarity, and ∇lag(x, y) almost = 0
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
function ncl(nlc::NLCModel, maxIt::Int64)
    # ** I. Preliminaries
        # ** I.1 Names and variables
            ρ_k = 1 # step
            τ = 10 # scale (used to update the ρ_k step)
            α = 0.5 # Constant (α needs to be < 1)
            β = 1 # Constant

            ω_end = 1 # global tolerance
            ω_k = 1 # sub problem tolerance
            η_end = 1 # global infeasability
            η_k = 2 # sub problem infeasability

            # initial points
            x_k = zeros(Float64, nlp_nvar, 1)
            y_k = zeros(Float64, nlp_ncon, 1)
            r_k = zeros(Float64, nlp_ncon, 1)
            z_k = zeros(Float64, nlp_nvar + nlp_ncon, 1)


    # ** II. Optimization loop
        k = 0
        converged = false
        while k < maxIt && !converged
            k += 1
            
            # ** II.1 Create the sub problem NC_k
                # X = vcat(x, r) (size(x) = nlc.nvar_x, size(r) = nlc.nvar_r)

                # ? (Complique) Nouveau x_0 choisi ici = ancienne solution. Demarrage a chaud...
                
                


            # ** II.2 Get subproblem's solution
                x_k, y_k, r_k, z_k = ipopt(nlc) # TODO: link with KNITRO/IPOPT
                # TODO (recherche) : Points intérieurs à chaud...
                # TODO (recherche) : tester la proximité des multiplicateurs de renvoyés par KNITRO et le y_k du problème (si r petit, probablement proches.)

            # ** II.3 Treatment & update
                if norm(r_k,Inf) <= max(η_k, η_end) # The residue has decreased enough
                    
                    y_k = y_k + ρ_k * r_k # Updating multiplicator
                    η_k = η_k / (1 + ρ_k ^ β) # (heuristic)
                    
                    # ** II.3.1 Solution found ?
                        converged = NLPModel_solved(nlc, x_k, y_k) # TODO (~recherche) : Voir si nécessaire ou si lorsque la tolérance de KNITRO renvoyée est assez faible et r assez petit, on a aussi résolu le problème initial
                
                else # The residue is to still too large
                    ρ_k = τ * ρ_k # increase the step # TODO (recherche) : Mieux choisir le pas pour avoir une meilleure convergence
                    η_k = η_end / (1 + ρ_k ^ α) # Change infeasability (heuristic) # ? (simple) η_end ou η_0, cf article
                    # TODO (recherche...) : update ω_k 
                end
                # ? Chez Nocedal & Wright, p.521, on a : ω_k = 1/ρ_k, ρ_k = 100ρ_k, η_k = 1/ρ_k^0.1
        end
    
    return x_k, y_k, r_k, z_k, converged # converged tells us if the solution returned is optimal or not
end