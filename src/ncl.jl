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

    #** I. Bounds constraints
        #** II.1 Feasability
            for i in 1:nlp.meta.nvar 
                if !(nlp.meta.lvar[i] - η <= x[i] <= nlp.meta.uvar[i] + η) 
                    if printing
                        println("    variable " * string(i) * " out of bounds + tolerance") 
                    end
                    return false
                end

        #** I.2 Complementarity for bounds
                    if nlp.meta.lvar[i] > -Inf # ? Optimiser avec jupp, jlow ?
                        if z_L[i] * (x[i] - nlp.meta.lvar[i]) > ϵ  
                            if printing
                                println("    complementarity not respected, see lowerbound of var " * string(i))
                                @show z_L[i] * (nlp.meta.lvar[i] - x[i])
                            end
                            return false
                        end
                    end
                    if nlp.meta.uvar[i] < Inf # ? Optimiser avec jupp, jlow ?
                        if z_U[i] * (nlp.meta.uvar[i] - x[i]) > ϵ  
                            if printing
                                println("    complementarity not respected, see upperbound of var " * string(i))
                                @show z_U[i] * (nlp.meta.uvar[i] - x[i])
                            end
                            return false
                        end
                    end
            end
    
    

    #** II. Other constraints
        c_x = cons(nlp, x)
        y_temp = y

        #** II.1 Feasability
            for i in 1:nlp.meta.ncon
                if !(nlp.meta.lcon[i] - η <= c_x[i] <= nlp.meta.ucon[i] + η)
                    if printing
                        println("    constraint " * string(i) * " out of bounds + tolerance")
                        println("    c_x[", i, "]               = ", c_x[i])
                        println("    nlp.meta.ucon[", i, "] + η = ", nlp.meta.ucon[i] + η)
                        println("    nlp.meta.lcon[", i, "] - η = ", nlp.meta.lcon[i] - η)
                    end
                    return false 
                end

                if i in nlp.meta.jinf
                    @warn "    infeasable problem passed to NLPModel_solved.\n    Check the constraint" * string(i)
                    return(false)
                end

        #** II.2 Complementarity
                if nlp.meta.lcon[i] > -Inf # ? Optimiser avec jupp, jlow ?
                    if !((i in nlp.meta.jrng) | (i in nlp.meta.jfix))
                        y_temp[i] = - abs(y_temp[i])
                        if printing
                            println("    y[", i, "]      = ", y[i])
                            println("    y_temp[", i, "] = ", y_temp[i])
                        end
                    end

                    if (y[i] * (c_x[i] - nlp.meta.lcon[i])) > ϵ
                        if printing
                            println("    complementarity not respected, see lower cons " * string(i))
                            println("    y[", i, "]             = ", y[i])
                            println("    c_x[", i, "]           = ", c_x[i])
                            println("    nlp.meta.lcon[", i, "] = ", nlp.meta.lcon[i])
                        end
                        return false
                    end
                end
            
                if nlp.meta.ucon[i] < Inf # ? Optimiser avec jupp, jlow ?
                    if !((i in nlp.meta.jrng) | (i in nlp.meta.jfix))
                        y_temp[i] = abs(y_temp[i])
                        if printing
                            println("    y[", i, "]      = ", y[i])
                            println("    y_temp[", i, "] = ", y_temp[i])
                        end
                    end

                    if (y[i] * (nlp.meta.ucon[i] - c_x[i])) > ϵ  
                        if printing
                            println("    complementarity not respected, see upper cons " * string(i))
                            println("    y[", i, "]             = ", y[i])
                            println("    c_x[", i, "]           = ", c_x[i])
                            println("    nlp.meta.ucon[", i, "] = ", nlp.meta.ucon[i])
                        end
                        return false
                    end

                end
            end
    
    
    #** III. Lagrangian
        ∇f_x = grad(nlp, x)       
        ∇lag = ∇f_x - jtprod(nlp, x, y_temp) + z_L - z_U
        
        if norm(∇lag, Inf) > ω # Not a stationnary point for the lagrangian
            if printing
                println("    not a stationnary point for the lagrangian")
                println("    ∇f_x              = ", ∇f_x)
                println("    t(Jac_x)          = ", transpose(jac(nlp, x)))
                println("    y_temp            = ", y_temp)
                println("    t(Jac_x) * y_temp = ", jtprod(nlp, x, y_temp))
                println("    -z_U              = ", -z_U)
                println("    z_L               = ", z_L)
                println("    ∇lag_x            = ", ∇lag)
            end
            return false
        end
    
    if printing
        println("    " * nlp.meta.name * " solved !")
    end
    return true # all the tests were passed, x, y respects feasability, complementarity not respected, see and ∇lag(x, y) almost = 0
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
        #\_end = global tolerance for complementarity conditions
        
        # initial points
        x_k = zeros(Type, nlc.nvar_x)
        r_k = zeros(Type, nlc.nvar_r)
        λ_k = zeros(Type, nlc.meta.ncon)
        z_k_U = zeros(Type, length(nlc.meta.uvar))
        z_k_L = zeros(Type, length(nlc.meta.lvar))

    # ** II. Optimization loop and return
        k = 0
        converged = false

        while k < max_iter && !converged
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
                    if printing_iterations
                        @show resolution_k.solution
                    end

                    x_k = resolution_k.solution.x[1:nlc.nvar_x]
                    r_k = resolution_k.solution.x[nlc.nvar_x+1:nlc.nvar_r]

                    # Get multipliers
                    λ_k = resolution_k.solver_specific[:multipliers_con]
                    z_k_U = resolution_k.solver_specific[:multipliers_U] #! =[] dans ce cas, pas séparé par KNITRO...
                    z_k_L = resolution_k.solver_specific[:multipliers_L]
                end
                if printing_iterations
                    println("   -----------------")
                    println("    k             = ", k, 
                            "\n    nlc.ρ         = ", nlc.ρ, 
                            "\n    η_k           = ", η_k, 
                            "\n    norm(r_k,Inf) = ", norm(r_k,Inf), 
                            "\n    x_k           = ", x_k, 
                            "\n    λ_k           = ", λ_k, 
                            "\n    nlc.y         = ", nlc.y, 
                            "\n    r_k           = ", r_k)
                end
                
                # TODO (recherche) : Points intérieurs à chaud...
                # TODO (recherche) : tester la proximité des multiplicateurs de renvoyés par KNITRO et le y_k du problème (si r petit, probablement proches.)

            # ** II.2 Treatment & update
                if norm(r_k,Inf) <= max(η_k, η_end) # The residue has decreased enough
                    nlc.y = nlc.y + nlc.ρ * r_k # Updating multiplier
                    η_k = η_k / (1 + nlc.ρ ^ β) # (heuristic)
                    
                    # ** II.2.1 Solution found ?
                    #tolerance
                    if (norm(r_k,Inf) <= min(η_k, η_end)) | (k == max_iter) # check if r_k is small enough, or if we've reached the end
                        if printing_iterations
                            println("    r_k small enough for optimality checking")
                        end

                        sol = vcat(x_k, r_k) # TODO: optimiser (cf resolution_k...)

                        ## Testing
                        converged = NLPModel_solved(nlc.nlp, x_k, -λ_k, z_k_U[1:nlc.nvar_x], z_k_L[1:nlc.nvar_x], ω_end, η_end, ϵ_end, printing_check) # TODO (~recherche) : Voir si nécessaire ou si lorsque la tolérance de KNITRO renvoyée est assez faible et r assez petit, on a aussi résolu le problème initial    
                        if printing_check & !converged # means we printed some thing with NLPModel_solved, so we skip a line
                            println("\n")
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
                                                        solution = sol,
                                                        iter = k,
                                                        objective=obj(nlc, sol), 
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
            
            if printing_iterations
                println("\n    k             = ", k, 
                        "\n    nlc.ρ         = ", nlc.ρ, 
                        "\n    η_k           = ", η_k, 
                        "\n    norm(r_k,Inf) = ", norm(r_k,Inf), 
                        "\n    x_k           = ", x_k, 
                        "\n    λ_k           = ", λ_k, 
                        "\n    nlc.y         = ", nlc.y, 
                        "\n    r_k           = ", r_k)
            end
        end    
end