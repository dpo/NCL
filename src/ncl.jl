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

    
    Checks if the nlp initial problem is solved with the x, y, z points withine the ω tolerance
    

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
    #? z = vcat(x, r) ???
    converged: a booleam, telling us if the progra; converged or reached the maximum of iterations fixed
"""

function KNITRO(nlp, ω)
    return [1,2],[2,1],[2,1],[1,2,2,1]
end

function ncl(nlp, maxIt::Int64)
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

            # Pre-access and sorting
            nlp_nvar = nlp.meta.nvar 
            nlp_ncon = nlp.meta.ncon 

            nlp_jfix = nlp.meta.jfix
            nlp_jlow = nlp.meta.jlow
            nlp_jupp = nlp.meta.jupp
            nlp_jrng = nlp.meta.jrng
            nlp_jfree = nlp.meta.jfree
            nlp_jinf = nlp.meta.jinf

            nlp_lvar = nlp.meta.lvar
            nlp_uvar = nlp.meta.uvar
            
            nlp_ucon = nlp.meta.ucon
            nlp_lcon = nlp.meta.lcon
            #@show [nlp_lcon[i] for i in 1:nlp_jfix], [nlp_lcon[i] for i in 1:nlp_jlow], [nlp_lcon[i] for i in 1:nlp_jupp], [nlp_lcon[i] for i in 1:nlp_jrng], [nlp_lcon[i] for i in 1:nlp_jfree], [nlp_lcon[i] for i in 1:nlp_jinf], 
            
            
            nlp_lcon = vcat([nlp_lcon[i] for i in nlp_jfix], # Sorting, useful for subproblems, see optimization loop
                            [nlp_lcon[i] for i in nlp_jlow], 
                            [nlp_lcon[i] for i in nlp_jupp], 
                            [nlp_lcon[i] for i in nlp_jrng], 
                            [nlp_lcon[i] for i in nlp_jfree], 
                            [nlp_lcon[i] for i in nlp_jinf]
                            )
            nlp_ucon = nlp.meta.ucon
            nlp_ucon = vcat([nlp_ucon[i] for i in nlp_jfix], # Sorting, useful for subproblems, see optimization loop
                            [nlp_ucon[i] for i in nlp_jlow], 
                            [nlp_ucon[i] for i in nlp_jupp], 
                            [nlp_ucon[i] for i in nlp_jrng], 
                            [nlp_ucon[i] for i in nlp_jfree], 
                            [nlp_ucon[i] for i in nlp_jinf]
                            )
            
            # ? (optimization) Probably not efficient at ADNLPModel
            # TODO (optimization) check efficiency !
            #nlp_cons(x) = vcat([cons(nlp, x)[i] for i in nlp_jfix], # Sorting, useful for subproblems, see optimization loop
            #                   [cons(nlp, x)[i] for i in nlp_jlow], 
            #                   [cons(nlp, x)[i] for i in nlp_jupp], 
            #                   [cons(nlp, x)[i] for i in nlp_jrng], 
            #                   [cons(nlp, x)[i] for i in nlp_jfree], 
            #                   [cons(nlp, x)[i] for i in nlp_jinf]
            #                   )


            # initial points
            x_k = zeros(Float64, nlp_nvar, 1)
            y_k = zeros(Float64, nlp_ncon, 1)
            r_k = zeros(Float64, nlp_ncon, 1)
            z_k = zeros(Float64, nlp_nvar + nlp_ncon, 1)


        # ** I.2 Test argument nlp consistency with hypothesis
            for i in 1:nlp_ncon
                if (nlp_lcon[i] != 0) | (nlp_ucon[i] != Inf)
                    println("ERROR: nlp argument should be in a standard form, like : 
                            min/max f(x)
                            subject to c(x) >= 0, Ax >= b, l <= x <= b
                            Try to change your lcon and ucon, at index " * string(i))
                    @show nlp_lcon
                    @show nlp_ucon
                    return Inf, Inf, Inf, Inf, false
                end
            end        


    # ** II. Optimization loop
        k = 0
        converged = false
        while k < maxIt && !converged
            k += 1
            
            # ** II.1 Create the sub problem NC_k
                # z = vcat(x, r) (size(x) = nvar, size(r) = ncon)

                c_k(z) = vcat([cons(nlp, z[1:nlp_nvar])[i] for i in nlp_jfix], # ? (Faisable) Comment traiter les contraintes d'egalite avec NCL ???
                              [cons(nlp, z[1:nlp_nvar])[i] + z[nlp_nvar + i] for i in nlp_jlow], # Supposed to contain every constraint, in the first case
                              [cons(nlp, z[1:nlp_nvar])[i] for i in nlp_jupp], # Assumed to be empty
                              [cons(nlp, z[1:nlp_nvar])[i] for i in nlp_jrng], # Assumed to be empty
                              [cons(nlp, z[1:nlp_nvar])[i] for i in nlp_jfree], # Assumed to be empty
                              [cons(nlp, z[1:nlp_nvar])[i] for i in nlp_jinf] # Assumed to be empty
                              ) # TODO (optimization): optimize this computation, not efficient with that much calls to cons()
                
                
                f_k(z) = obj(nlp, z[1:nlp_nvar]) + # Objective function of the initial problem
                         (y_k' * z[nlp_nvar+1 : end])[1] + # Lagrangian part, using residue r # "[1]" is just to avoid an error because of 1 + [0]. It converts it into 1 + 0.
                         0.5 * ρ_k * norm(z[nlp_nvar+1 : end], 2) ^ 2 # Augmented part, residue as well
                

#+(::ForwardDiff.Dual{ForwardDiff.Tag{getfield(Main, Symbol("#f_k#686")){ADNLPModel,Int64},Int64},ForwardDiff.Dual{ForwardDiff.Tag{getfield(Main, Symbol("#f_k#686")){ADNLPModel,Int64},Int64},Int64,2},2}, 
#  ::Array{ForwardDiff.Dual{ForwardDiff.Tag{getfield(Main, Symbol("#f_k#686")){ADNLPModel,Int64},Int64},ForwardDiff.Dual{ForwardDiff.Tag{getfield(Main, Symbol("#f_k#686")){ADNLPModel,Int64},Int64},Int64,2},2},1})

                @show f_k([1,0,0])
                # ? (Complique) Nouveau x_0 choisi ici = ancienne solution. Demarrage a chaud...
                
                NC_k = ADNLPModel(f_k, z_k ; lvar = nlp_lvar, uvar = nlp_uvar, c = c_k) # Sub problem modelisation
                # TODO (debug): checker que les bornes des contraintes et les contraintes sont dans le même ordre (print...)

            # ** II.2 Get subproblem's solution
                x_k, y_k, r_k, z_k = KNITRO(NC_k, ω_k) # TODO: link with KNITRO/IPOPT
                # TODO (recherche) : Points intérieurs à chaud...

            # ** II.3 Treatment & update
                if norm(r_k,Inf) <= max(η_k, η_end) # The residue has decreased enough
                    
                    y_k = y_k + ρ_k * r_k # Updating multiplicator
                    η_k = η_k / (1 + ρ_k ^ β) # (heuristic)
                    
                    # ** II.3.1 Solution found ?
                        ∇f_x = grad(nlp, x_k)
                        c_x = cons(nlp, x_k)

                        feasable = true # by default, x_k is a feasable point. Then we check with the constraints
                        optimal = true # same, we will check with KKT conditions

                    # ? (simple) Peut-être pas necessaire, KNITRO/IPOPT renvoie probablement une solution realisable
                        for i in 1:nlp.nvar 
                            if !(nlp_lvar[i] <= x_k[i] <= nlp_uvar[i]) # bounds constraints
                                feasable = false # bounds constraints not respected
                                break
                            end
                        end

                        if feasable
                            for i in 1:nlp.ncon
                                if !(nlp_lcon[i] <= c_x[i] <= nlp_ucon[i]) # other constraints
                                    feasable = false # bounds constraints not respected
                                    break
                                end
                            end

                            if feasable
                                grad_lag = ∇f_x - jtprod(nlp, x_k, y_k) # ? (debug) Transposee, ou pas, verifier

                                if norm(grad_lag, Inf) > ω_end
                                    optimal = false
                                end
                            end
                        end

                        converged = feasable & optimal
                
                else # The residue is to still too large
                    ρ_k = τ * ρ_k # increase the step # TODO (recherche) : Mieux choisir le pas pour avoir une meilleure convergence
                    η_k = η_end / (1 + ρ_k ^ α) # Change infeasability (heuristic) # ? (simple) η_end ou η_0, cf article
                end
        end
    
    return x_k, y_k, r_k, z_k, converged # converged tells us if the solution returned is optimal or not
end