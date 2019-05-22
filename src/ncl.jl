# comment
# ** Important
# ! Alert, problem
# ? Question
# TODO 

using LinearAlgebra
using NLPModels

"""
# ! A eviter car recopie nlp, perd beaucoup de temps
# ? Semblant de pointeur possible ? Mais appel a la fonction long aussi. Inline faisable ?

    Checks if the nlp initial problem is solved with the x, y, z points withine the ω tolerance
    
    function nc_k(nlp, y_k) # TODO
    end

    
    Checks if the nlp initiql problem is solved with the x, y, z points withine the ω tolerance
    

    function NCO_solved(x, y, z, ω, nlp) # TODO
        return(true)
    end

"""

"""
NCL method implementation. See https://www.researchgate.net/publication/325480151_Stabilized_Optimization_Via_an_NCL_Algorithm for further explications on the method
Arguments: 
    - nlp: optimization problem described by the modelization NLPModels.jl (voir https://github.com/JuliaSmoothOptimizers/NLPModels.jl)
      nlp is the generic problem you want to solve
Returns:
    x: position of the optimum found
    y: optimal lagrangian multiplicator
    r: value of constraints (around 0 if converged)
    z: # ? C'est qui z finalement ?
    converged: a booleam, telling us if the progra; converged or reached the maximum of iterations fixed
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

        x_k, y_k, r_k, z_k = 0, 0, 0, 0 # initial points
        NC_k = nlp # sub problem

        # constraints (pre-access)
        lvar = nlp.lvar
        uvar = nlp.uvar
        lcon = nlp.lcon
        ucon = nlp.ucon

    # ** II. Optimization loop
        k = 0
        converged = false
        while k < maxIt && !converged
            k += 1
            
            # ** II.1 Create the sub problem NC_k  
                NC_k = nc_k(nlp, y_k) # Sub problem modelisation # TODO 
            
            # ** II.2 Get subproblem's solution
                x_k, y_k, r_k, z_k = KNITRO(NC_k, ω_k) # TODO: link with KNITRO/IPOPT
                # TODO (loin) : Points intérieurs à chaud...

            # ** II.3 Treatment & update
                if norm(r_k,Inf) <= max(η_k, η_end) # The residue has decreased enough
                    
                    y_k = y_k + ρ_k * r_k # Updating multiplicator
                    η_k = η_k / (1 + ρ_k ^ β) # (heuristic)
                    
                    # ** II.3.1 Solution found ?
                        ∇fx = grad(nlp, x_k)
                        cx = cons(nlp, x_k)
                        Jcx = jac(nlp, x_k)

                        feasable = true # by default, x_k is a feasable point. Then we check with the constraints
                        optimal = true # same, we will check with KKT conditions

                    # ? Peut-être pas necessaire, KNITRO/IPOPT doit renvoyer une solution realisable
                        for i in 1:nlp.nvar 
                            if !(lvar[i] <= x_k[i] <= uvar[i]) # bounds constraints
                                feasable = false # bounds constraints not respected
                                break
                            end
                        end

                        if feasable
                            for i in 1:nlp.ncon
                                if !(lcon[i] <= cx[i] <= ucon[i]) # other constraints
                                    feasable = false # bounds constraints not respected
                                    break
                                end
                            end

                            if feasable
                                grad_lag = ∇fx - jprod(nlp, x_k, y_k) # ? Transposee ?

                                if norm(grad_lag, Inf) > ω_end
                                    optimal = false
                                end
                            end
                        end

                        converged = feasable && optimal
                
                else # The residue is to still too large
                    ρ_k = τ * ρ_k # increase the step # TODO (loin) : Mieux choisir le pas pour avoir une meilleure convergence
                    η_k = η_end / (1 + ρ_k ^ α) # Change infeasability (heuristic) # ? η_end ou η_0, cf article
                end
        end
    
    return( x_k, y_k, r_k, z_k, converged) # converged tells us if the solution returned is optimal or not
end