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

# TODOs :
    #! si r_k = [], on r2sout toujours au même endroit, envoyer à Ipopt
    #! erreur dans \nabla_lag .......y_temp.......



"""
Tests if the (x, y) is a solution of the KKT conditions of the nlp problem (nlp follows the NLPModels.jl formalism) within ω as a tolerance
Note: the lagrangian is considered as :
    l(x, y) = f(x) - y' * c(x)          (!!! -, not + !!!)
"""
function NLPModel_solved(nlp::AbstractNLPModel, x::Vector{<:Real}, y::Vector{<:Real}, z_U::Vector{<:Real}, z_L::Vector{<:Real}, ω::Real, η::Real, ϵ::Real, printing::Bool) ::Bool
    if printing
        println("\nNLPModel_solved called on " * nlp.meta.name)
    end

    #** 0. Warnings and mode determination
        if any(z_U .< -ϵ) & any(z_L .> ϵ) 
            println("    z_U = ", z_U)
            println("    z_L = ", z_L)

            error("sign problem of z_U or z_L passed in argument to NLPModel_solved.\n 
                z_U and z_L are supposed to be of same sign : z_U >= 0, z_L <= 0.\n 
                Here, some components of z_U or z_L are of the wrong sign")
        end

        if (z_U == []) & (nlp.meta.iupp != []) # NLPModelsKnitro returns z_U = [], "didn't find how to treat those separately"
            knitro = true
            z = z_L
            z_temp = copy(z)
        else 
            knitro = false
        end
    
    #** I. Bounds constraints
        if knitro
            #** II.1 Feasability (same if knitro or not)
                for i in 1:nlp.meta.nvar 
                    if !(nlp.meta.lvar[i] - η <= x[i] <= nlp.meta.uvar[i] + η) 
                        if printing
                            println("    variable " * string(i) * " out of bounds + tolerance") 
                            println("    x[, ", i, "] = ", x[i])
                            println("    nlp.meta.lvar[, ", i, "] = ", nlp.meta.lvar[i])
                            println("    nlp.meta.uvar[, ", i, "] = ", nlp.meta.uvar[i])
                        end
                        return false
                    end
                end

            #** [knitro] I.2 Complementarity for bounds
                #* [knitro] I.2.1 Lower bound complementarity
                    for i in nlp.meta.ilow
                        if z[i] * (x[i] - nlp.meta.lvar[i]) > ϵ  
                            if printing
                                println("    complementarity = ", (z[i] * (x[i] - nlp.meta.lvar[i])), " out of tolerance ϵ = ", ϵ, ". See lower var cons " * string(i))
                                println("      z[", i, "]             = ", z[i])
                                println("      x[", i, "]           = ", x[i])
                                println("      nlp.meta.lvar[", i, "] = ", nlp.meta.lvar[i])
                            end
                            return false
                        end
                    end
                        

                #* [knitro] I.2.1 Upper bound complementarity
                    for i in nlp.meta.iupp
                        if z[i] * (x[i] - nlp.meta.uvar[i]) > ϵ  
                            if printing
                                println("    complementarity = ", (z[i] * (x[i] - nlp.meta.uvar[i])), " out of tolerance ϵ = ", ϵ, ". See upper var cons " * string(i))
                                println("      z[", i, "]             = ", z[i])
                                println("      x[", i, "]           = ", x[i])
                                println("      nlp.meta.uvar[", i, "] = ", nlp.meta.uvar[i])
                            end
                            return false
                        end
                    end
        else
            #** II.1 Feasability (same if knitro or not)
                for i in 1:nlp.meta.nvar 
                    if !(nlp.meta.lvar[i] - η <= x[i] <= nlp.meta.uvar[i] + η) 
                        if printing
                            println("    variable " * string(i) * " out of bounds + tolerance") 
                            println("    x[, ", i, "] = ", x[i])
                            println("    nlp.meta.lvar[, ", i, "] = ", nlp.meta.lvar[i])
                            println("    nlp.meta.uvar[, ", i, "] = ", nlp.meta.uvar[i])
                        end
                        return false
                    end
        
            #** [usual] I.2 Complementarity for bounds
                #* [usual] I.2.1 Lower bound complementarity
                    if nlp.meta.lvar[i] > -Inf
                        if z_L[i] * (x[i] - nlp.meta.lvar[i]) > ϵ  
                            if printing
                                println("    complementarity = ", (z_L[i] * (x[i] - nlp.meta.lvar[i])), " out of tolerance ϵ = ", ϵ, ". See lower var cons " * string(i))
                                println("      z_L[", i, "]             = ", z_L[i])
                                println("      x[", i, "]           = ", x[i])
                                println("      nlp.meta.lvar[", i, "] = ", nlp.meta.lvar[i])
                            end
                            return false
                        end
                    end
    
                #* [usual] I.2.1 Upper bound complementarity
                    if nlp.meta.uvar[i] < Inf
                        if z_U[i] * (x[i] - nlp.meta.uvar[i]) > ϵ  
                            if printing
                                println("    complementarity = ", (z_U[i] * (x[i] - nlp.meta.uvar[i])), " out of tolerance ϵ = ", ϵ, ". See upper var cons " * string(i))
                                println("      z_U[", i, "]             = ", z_U[i])
                                println("      x[", i, "]           = ", x[i])
                                println("      nlp.meta.uvar[", i, "] = ", nlp.meta.uvar[i])
                            end
                            return false
                        end
                    end
                end
        end
    
    

    #** II. Other constraints
        #** II.0 Precomputation
            c_x = cons(nlp, x) # Precomputation
            y_temp = copy(y) # real copy, to avoid initial data modification

        #** II.1 Feasability
            for i in 1:nlp.meta.ncon
                if !(nlp.meta.lcon[i] - η <= c_x[i] <= nlp.meta.ucon[i] + η)
                    if printing
                        println("    constraint " * string(i) * " out of bounds + tolerance")
                        println("      c_x[", i, "]               = ", c_x[i])
                        println("      nlp.meta.ucon[", i, "] + η = ", nlp.meta.ucon[i] + η)
                        println("      nlp.meta.lcon[", i, "] - η = ", nlp.meta.lcon[i] - η)
                    end
                    return false 
                end

                if i in nlp.meta.jinf
                    @warn "    infeasable problem passed to NLPModel_solved.\n    Check the constraint" * string(i)
                    return false
                end
            end

        #** II.2 Complementarity
            #* II.2.1 Lower complementarity
                for i in nlp.meta.jlow # lower constraints
                    if (y_temp[i] >= ϵ)
                        y_temp[i] = - y_temp[i]
                        if printing
                            println("    y[", i, "]      = ", y[i])
                            println("    y_temp[", i, "] = ", y_temp[i])
                        end
                    end

                    if !(-ϵ <= (y_temp[i] * (c_x[i] - nlp.meta.lcon[i])) <= ϵ) # Complemntarity condition (sign handled by the condition above and the feasability test)
                        if printing
                            println("    complementarity = ", (y_temp[i] * (c_x[i] - nlp.meta.lcon[i])), " out of tolerance ϵ = ", ϵ, ". See lower cons " * string(i))
                            println("      y[", i, "]             = ", y[i])
                            println("      c_x[", i, "]           = ", c_x[i])
                            println("      nlp.meta.lcon[", i, "] = ", nlp.meta.lcon[i])
                        end
                        return false
                    end
                end
            
            #* II.2.1 Upper complementarity
                for i in nlp.meta.jupp # upper constraints
                    if y_temp[i] <= -ϵ
                        y_temp[i] = - y_temp[i]
                        if printing
                            println("    y[", i, "]      = ", y[i])
                            println("    y_temp[", i, "] = ", y_temp[i])
                        end
                    end

                    if !(-ϵ <= (y_temp[i] * (c_x[i] - nlp.meta.ucon[i])) <= ϵ)  # Complmentarity condition (sign handled by the condition above and the feasability test)
                        if printing
                            println("    complementarity = ", (y_temp[i] * (c_x[i] - nlp.meta.ucon[i])), " out of tolerance ϵ = ", ϵ, ". See upper cons " * string(i))
                            println("      y[", i, "]             = ", y[i])
                            println("      c_x[", i, "]           = ", c_x[i])
                            println("      nlp.meta.ucon[", i, "] = ", nlp.meta.ucon[i])
                        end
                        return false
                    end

                end
    
    
    #** III. Lagrangian
        ∇f_x = grad(nlp, x)      
        
        if knitro
            ∇lag_x = ∇f_x + jtprod(nlp, x, y_temp) + z_L
            #∇lag_x = ∇f_x - jtprod(nlp, x, y) + z_L

            if norm(∇lag_x, Inf) > ω # Not a stationnary point for the lagrangian
                if printing
                    println("    not a stationnary point for the lagrangian")
                    println("      ∇f_x              = ", ∇f_x)
                    println("      t(Jac_x)          = ", transpose(jac(nlp, x)))
                    println("      y_temp            = ", y_temp)
                    println("      y                 = ", y)
                    println("      t(Jac_x) * y_temp = ", jtprod(nlp, x, y_temp))
                    println("      t(Jac_x) * y      = ", jtprod(nlp, x, y))
                    println("      z                 = ", z)
                    println("      ∇lag_x            = ", ∇lag_x)
                end
                return false
            end
        else
            ∇lag_x = ∇f_x + jtprod(nlp, x, y_temp) - z_L - z_U
            #∇lag_x = ∇f_x + jtprod(nlp, x, y) + z_L - z_U



            #! Bizarre
            #? - z_L - z_U fonctionne...
            #! Bizarre







            if norm(∇lag_x, Inf) > ω # Not a stationnary point for the lagrangian
                if printing
                    println("    not a stationnary point for the lagrangian")
                    println("      ∇f_x              = ", ∇f_x)
                    println("      t(Jac_x)          = ", transpose(jac(nlp, x)))
                    println("      y_temp            = ", y_temp)
                    println("      y                 = ", y)
                    println("      t(Jac_x) * y_temp = ", jtprod(nlp, x, y_temp))
                    println("      t(Jac_x) * y      = ", jtprod(nlp, x, y))
                    println("      -z_U              = ", -z_U)
                    println("      z_L               = ", z_L)
                    println("      ∇lag_x            = ", ∇lag_x)
                end
                return false
            end
        end
        
        
    
    if printing
        println("    " * nlp.meta.name * " solved !")
    end
    return true # all the tests were passed, x, y respects feasability, complementarity not respected, see and ∇lag_x(x, y) almost = 0
end


"""
NCL method implementation. See https://www.researchgate.net/publication/325480151_Stabilized_Optimization_Via_an_NCL_Algorithm for further explications on the method
Arguments: 
    - nlp: optimization problem described by the modelization NLPModels.jl (voir https://github.com/JuliaSmoothOptimizers/NLPModels.jl)
      nlp is the generic problem you want to solve
Returns:
    a GenericExecutionStats, based on the NLPModelsIpopt/Knitro return :
        SolverTools.status                              # of the last resolution
        nlc                                             # the problem in argument,
        solution = sol,                                 # solution found
        iter = k,                                       # number of iteration of the ncl method (not iteration to solve subproblems)
        objective=obj(nlc, sol),                        # objective value
        elapsed_time=0,                                 # time of computation of the whole resolution
        solver_specific=Dict(:multipliers_con => λ_k,   # lagrangian multipliers for : constraints
                            :multipliers_L => z_k_L,    #                              upper bounds
                            :multipliers_U => z_k_U     #                              lower bounds
                            )
        )
"""

#! Problem : Type error si pas de convergence


function NCLSolve(nlc::NLCModel, max_iter::Int64, use_ipopt::Bool, ω_end::Real, η_end::Real, ϵ_end::Real, printing_iterations::Bool, printing_iterations_solver::Bool, printing_check::Bool) ::GenericExecutionStats 
    if printing_iterations
        println("NCLSolve called on " * nlc.meta.name)
    end
    
    # ** I. Names and variables
        Type = typeof(nlc.meta.x0[1])
        nlc.ρ = 1.0 # step
        τ = 10.0 # scale (used to update the ρ_k step)
        α = 0.1 # Constant (α needs to be < 1)
        β = 0.2 # Constant

        # ω_end = global tolerance, in argument
        ω_k = 0.5 # sub problem tolerance
        #η_end = global infeasability in argument
        η_k = 2.0 # sub problem infeasability
        #ϵ_end = global tolerance for complementarity conditions
        
        # initial points
        x_k = zeros(Type, nlc.nvar_x)
        r_k = zeros(Type, nlc.nvar_r)
        λ_k = zeros(Type, nlc.meta.ncon)
        z_k_U = zeros(Type, length(nlc.meta.uvar))
        z_k_L = zeros(Type, length(nlc.meta.lvar))

    # ** II. Optimization loop and return
        k = 0
        converged = false

        while (k < max_iter) & !converged
            k += 1
            # ** II.1 Get subproblem's solution
                if use_ipopt
                        resolution_k = NLPModelsIpopt.ipopt(nlc, tol = ω_k, constr_viol_tol = η_k, compl_inf_tol = ϵ_end, print_level = printing_iterations_solver ? 3 : 0, ignore_time = true)
                        
                        # Get variables
                        x_k = resolution_k.solution[1:nlc.nvar_x]
                        r_k = resolution_k.solution[nlc.nvar_x+1 : nlc.nvar_x+nlc.nvar_r]

                        # Get multipliers
                        λ_k = resolution_k.solver_specific[:multipliers_con] 
                        z_k_U = resolution_k.solver_specific[:multipliers_U]
                        z_k_L = resolution_k.solver_specific[:multipliers_L]

                else # Knitro
                    resolution_k = _knitro(nlc)::GenericExecutionStats

                    # Get variables
                    x_k = resolution_k.solution.x[1:nlc.nvar_x]
                    r_k = resolution_k.solution.x[nlc.nvar_x+1:nlc.nvar_r]

                    # Get multipliers
                    λ_k = resolution_k.solver_specific[:multipliers_con]
                    z_k_U = resolution_k.solver_specific[:multipliers_U] #! =[] dans ce cas, pas séparé par KNITRO...
                    z_k_L = resolution_k.solver_specific[:multipliers_L]
                end

                if printing_iterations
                    println("   ----- Iter k = ", k, "-----",
                            "\n    nlc.ρ         = ", nlc.ρ, 
                            "\n    η_k           = ", η_k, 
                            "\n    norm(r_k,Inf) = ", norm(r_k,Inf), 
                            "\n    x_k           = ", x_k, 
                            "\n    λ_k           = ", λ_k, 
                            "\n    nlc.y         = ", nlc.y, 
                            "\n    r_k           = ", r_k)
                end
                
        # TODO (recherche) : Points intérieurs à chaud...
        # TODO (recherche) : tester la proximité des multiplicateurs λ_k de renvoyés par le solveur et le nlc.y du problème (si r petit, probablement proches.)

            # ** II.2 Treatment & update
                if norm(r_k,Inf) <= max(η_k, η_end) | (k == max_iter) # The residue has decreased enough
                    nlc.y = nlc.y + nlc.ρ * r_k # Updating multiplier
                    η_k = η_k / (1 + nlc.ρ ^ β) # (heuristic)
                    
                    # ** II.2.1 Solution found ?
                        if (norm(r_k,Inf) <= η_end) | (k == max_iter) # check if r_k is small enough, or if we've reached the end
                            if printing_iterations
                                println("    r_k small enough for optimality checking")
                            end

                            ## Testing
                            converged = NLPModel_solved(nlc.nlp, x_k, -λ_k, z_k_U[1:nlc.nvar_x], z_k_L[1:nlc.nvar_x], ω_end, η_end, ϵ_end, printing_check) # TODO (~recherche) : Voir si nécessaire ou si lorsque la tolérance de KNITRO renvoyée est assez faible et r assez petit, on a aussi résolu le problème initial    
                            if printing_check & !converged & printing_check # means we printed some thing with NLPModel_solved, so we skip a line
                                print("\n ------- Not fitting with KKT conditions ----------\n")
                            end
                            
                            status = resolution_k.status

                            if printing_iterations
                                if converged
                                    println("    EXIT: optimal solution found")
                                end
                                if k == max_iter
                                    println("    EXIT: reached max_iter")
                                end
                            end
                            
                            ## And return
                            if converged | (k == max_iter) # TODO: clarify
                                return GenericExecutionStats(status, nlc,
                                                            solution = resolution_k.solution,
                                                            iter = k,
                                                            objective=obj(nlc, resolution_k.solution), 
                                                            elapsed_time=0,
                                                            solver_specific=Dict(:multipliers_con => λ_k,
                                                                                :multipliers_L => z_k_L,
                                                                                :multipliers_U => z_k_U
                                                                                )
                                                            )
                            end
                        end

                else # The residue is to still too large
                    nlc.ρ = τ * nlc.ρ # increase the step # TODO (recherche) : Mieux choisir le pas pour avoir une meilleure convergence
                    η_k = η_end / (1 + nlc.ρ ^ α) # Change infeasability (heuristic) # ? (simple) η_end ou η_0, cf article
                    # TODO (recherche...) : update ω_k 
                end
                # ? Chez Nocedal & Wright, p.521, on a : ω_k = 1/nlc.ρ, nlc.ρ = 100ρ_k, η_k = 1/nlc.ρ^0.1
        end    
end