# comment
# ** Important
# ! Warning / Problem
# ? Question
# TODO 

using LinearAlgebra
using NLPModels
using SolverTools
using Ipopt
using NLPModelsIpopt
include("NCLModel.jl")

#using NLPModelsKnitro



# TODOs :
    #! si r_k = [], on resout toujours au même endroit, envoyer à Ipopt



"""
mult_format_check verifys that z_U and z_L are given in the right format to KKT_check. 
For the constraint multipliers, λ, if i is in jlow or jupp, λ[i] is modified to be >= 0 (it is done inside KKT_check). 

!!! Important note !!! The convention is : 
(P) min f(x)
    s.t. c(x) >= 0

And then
    multipliers λ >= 0
    Lagrangien(x, λ) = f(x) - λ' * c(x)
    ∇_{x}[lag(x, λ)] = ∇_{x}[f(x)] - t(Jac_{c(x)}) * λ - (z_L - z_U)
"""
function mult_format_check(z_U::Vector{<:Real}, z_L::Vector{<:Real}, ϵ::Real) ::Tuple{Vector{<:Real}, Vector{<:Real}}
    if (any(z_U .< -ϵ) & any(z_U .> ϵ))
        println("    z_U = ", z_U)

        error("sign problem of z_U passed in argument to KKT_check.
               Multipliers are supposed to be >= 0.
               Here, some components are negatives")
    end

    if (any(z_L .< -ϵ) & any(z_L .> ϵ))
        println("    z_L = ", z_L)

        error("sign problem of z_L passed in argument to KKT_check. 
               Multipliers are supposed to be >= 0.
               Here, some components are negatives")
    end

    if all(z_U .< ϵ) & any(z_U .< - ϵ)
        @warn "z_U was <= ϵ (complementarity tolerance) and non zero so it was changed to its opposite. Multipliers are supposed to be all >= 0"
        z_U = - z_U
    end

    if all(z_L .< ϵ) & any(z_L .< - ϵ)
        @warn "z_L was <= ϵ (complementarity tolerance) and non zero so it was changed to its opposite. Multipliers are supposed to be all >= 0"
        z_L = - z_L
    end

    return z_U, z_L
end



# TODO : Simplifier les complementarités, tout considérer comme un range
"""
KKT_check tests if (x, λ) is a solution of the KKT conditions of the nlp problem (nlp follows the NLPModels.jl formalism, it is suposed to be an AbstractNLPModel), within 
    ω as a tolerance for the lagrangian gradient norm
    η as a tolerance for constraint infeasability
    ϵ as a tolerance for complementarity checking

!!! Important note !!! the lagrangian is considered as :
    l(x, λ) = f(x) - λ' * c(x)          
    with c(x) >= 0
            λ >= 0
"""
function KKT_check(nlp::AbstractNLPModel, x::Vector{<:Real}, λ::Vector{<:Real}, z_U::Vector{<:Real}, z_L::Vector{<:Real}, ω::Real, η::Real, ϵ::Real, print_level::Int64) ::Bool
    if print_level >= 1
        println("\nKKT_check called on " * nlp.meta.name)
    end

    #** 0. Warnings and mode determination
        z_U, z_L = mult_format_check(z_U, z_L, ϵ)

        if (z_U == []) & (nlp.meta.iupp != []) # NLPModelsKnitro returns z_U = [], "didn't find how to treat those separately"
            knitro = true
            z = z_L
            z_temp = copy(z) #! Useful ?
        else 
            knitro = false
        end
    
    #** I. Bounds constraints
        if knitro
            #** II.1 Feasability (same if knitro or not)
                for i in 1:nlp.meta.nvar 
                    if !(nlp.meta.lvar[i] - η <= x[i] <= nlp.meta.uvar[i] + η) 
                        if print_level >= 2
                            println("    variable " * string(i) * " out of bounds + tolerance") 
                            
                            if print_level >= 3
                                println("      x[, ", i, "] = ", x[i])
                                println("      nlp.meta.lvar[, ", i, "] = ", nlp.meta.lvar[i])
                                println("      nlp.meta.uvar[, ", i, "] = ", nlp.meta.uvar[i])
                            end
                        end
                        return false
                    end
                end

            #** [knitro] I.2 Complementarity for bounds
                #* [knitro] I.2.1 Lower bound complementarity
                    for i in nlp.meta.ilow
                        if !(-ϵ <= z[i] * (x[i] - nlp.meta.lvar[i]) <= ϵ) # Complementarity condition (sign handled by the feasability test)
                            if print_level >= 2
                                println("    complementarity = ", (z[i] * (x[i] - nlp.meta.lvar[i])), " out of tolerance ϵ = ", ϵ, ". See lower var cons " * string(i))
                                
                                if print_level >= 3
                                    println("      z[", i, "]             = ", z[i])
                                    println("      x[", i, "]           = ", x[i])
                                    println("      nlp.meta.lvar[", i, "] = ", nlp.meta.lvar[i])
                                end
                            end
                            return false
                        end
                    end
                        

                #* [knitro] I.2.1 Upper bound complementarity
                    for i in nlp.meta.iupp
                        if !(-ϵ <= z[i] * (x[i] - nlp.meta.uvar[i]) <= ϵ) # Complementarity condition (sign handled by the feasability test)
                            if print_level >= 2
                                println("    complementarity = ", (z[i] * (x[i] - nlp.meta.uvar[i])), " out of tolerance ϵ = ", ϵ, ". See upper var cons " * string(i))

                                if print_level >= 3
                                    println("      z[", i, "]             = ", z[i])
                                    println("      x[", i, "]           = ", x[i])
                                    println("      nlp.meta.uvar[", i, "] = ", nlp.meta.uvar[i])
                                end
                            end
                            return false
                        end
                    end
        else
            #** II.1 Feasability (same if knitro or not)
                for i in 1:nlp.meta.nvar 
                    if !(nlp.meta.lvar[i] - η <= x[i] <= nlp.meta.uvar[i] + η) 
                        if print_level >= 2
                            println("    variable " * string(i) * " out of bounds + tolerance") 

                            if print_level >= 3
                                println("      x[, ", i, "] = ", x[i])
                                println("      nlp.meta.lvar[, ", i, "] = ", nlp.meta.lvar[i])
                                println("      nlp.meta.uvar[, ", i, "] = ", nlp.meta.uvar[i])
                            end
                        end
                        return false
                    end
        
            #** [usual] I.2 Complementarity for bounds
                #* [usual] I.2.1 Lower bound complementarity
                    if nlp.meta.lvar[i] > -Inf
                        if !(-ϵ <= z_L[i] * (x[i] - nlp.meta.lvar[i]) <= ϵ)
                            if print_level >= 2
                                println("    complementarity = ", (z_L[i] * (x[i] - nlp.meta.lvar[i])), " out of tolerance ϵ = ", ϵ, ". See lower var cons " * string(i))

                                if print_level >= 3
                                    println("      z_L[", i, "]             = ", z_L[i])
                                    println("      x[", i, "]           = ", x[i])
                                    println("      nlp.meta.lvar[", i, "] = ", nlp.meta.lvar[i])
                                end
                            end
                            return false
                        end
                    end
    
                #* [usual] I.2.1 Upper bound complementarity
                    if nlp.meta.uvar[i] < Inf
                        if !(-ϵ <= z_U[i] * (x[i] - nlp.meta.uvar[i]) <= ϵ)
                            if print_level >= 2
                                println("    complementarity = ", (z_U[i] * (x[i] - nlp.meta.uvar[i])), " out of tolerance ϵ = ", ϵ, ". See upper var cons " * string(i))

                                if print_level >= 3
                                    println("      z_U[", i, "]             = ", z_U[i])
                                    println("      x[", i, "]           = ", x[i])
                                    println("      nlp.meta.uvar[", i, "] = ", nlp.meta.uvar[i])
                                end
                            end
                            return false
                        end
                    end
                end
        end
    

    #** II. Other constraints
        #** II.0 Precomputation
            c_x = cons(nlp, x) # Precomputation
            λ_temp = copy(λ) # real copy, to avoid initial data modification #! Utile ?...
            

        #** II.1 Feasability
            for i in 1:nlp.meta.ncon
                if !(nlp.meta.lcon[i] - η <= c_x[i] <= nlp.meta.ucon[i] + η)
                    if print_level >= 2
                        println("    constraint " * string(i) * " out of bounds + tolerance")

                        if print_level >= 3
                            println("      c_x[", i, "]               = ", c_x[i])
                            println("      nlp.meta.ucon[", i, "] + η = ", nlp.meta.ucon[i] + η)
                            println("      nlp.meta.lcon[", i, "] - η = ", nlp.meta.lcon[i] - η)
                        end
                    end
                    return false 
                end

                if i in nlp.meta.jinf
                    error("    infeasable problem passed to KKT_check.\n    Check the constraint" * string(i))
                    return false
                end
            end

        #** II.2 Complementarity
            #* II.2.1 Lower complementarity
                for i in nlp.meta.jlow # lower constraints
                    if (λ_temp[i] <= - ϵ)
                        λ_temp[i] = - λ_temp[i]
                        @warn "λ[" * string(i) * "] = " * string(λ[i]) * " was <= -ϵ = " * string(-ϵ) * " so it was changed in its opposite. Check your convention for multipliers"
                    end

                    if !(-ϵ <= (λ_temp[i] * (c_x[i] - nlp.meta.lcon[i])) <= ϵ) # Complementarity condition (sign handled by the condition above and the feasability test)
                        if print_level >= 2
                            println("    complementarity = ", (λ_temp[i] * (c_x[i] - nlp.meta.lcon[i])), " out of tolerance ϵ = ", ϵ, ". See lower cons " * string(i))

                            if print_level >= 3
                                println("      λ[", i, "]             = ", λ[i])
                                println("      c_x[", i, "]           = ", c_x[i])
                                println("      nlp.meta.lcon[", i, "] = ", nlp.meta.lcon[i])
                            end
                        end
                        return false
                    end
                end
            
            #* II.2.2 Upper complementarity
                for i in nlp.meta.jupp # upper constraints
                    if (λ_temp[i] <= -ϵ)
                        λ_temp[i] = - λ_temp[i]
                        @warn "λ[" * string(i) * "] = " * string(λ[i]) * " was <= ϵ = " * string(-ϵ) * " so it was changed in its opposite. Check your convention for multipliers"
                    end

                    if !(-ϵ <= (λ_temp[i] * (c_x[i] - nlp.meta.ucon[i])) <= ϵ)  # Complmentarity condition (sign handled by the condition above and the feasability test)
                        if print_level >= 2
                            println("    complementarity = ", (λ_temp[i] * (c_x[i] - nlp.meta.ucon[i])), " out of tolerance ϵ = ", ϵ, ". See upper cons " * string(i))

                            if print_level >= 3
                                println("      λ[", i, "]             = ", λ[i])
                                println("      c_x[", i, "]           = ", c_x[i])
                                println("      nlp.meta.ucon[", i, "] = ", nlp.meta.ucon[i])
                            end
                        end
                        return false
                    end

                end

            #* II.2.3 Range complementarity 
                for i in nlp.meta.jrng # upper constraints
                    if !(-ϵ <= (λ_temp[i] * (c_x[i] - nlp.meta.ucon[i])) <= ϵ) & !(-ϵ <= (λ_temp[i] * (c_x[i] - nlp.meta.lcon[i])) <= ϵ)  # Complementarity condition (for range constraint, we have necessarily : [λ_temp[i] * (c_x[i] - nlp.meta.lcon[i])] * [λ_temp[i] * (c_x[i] - nlp.meta.ucon[i])] = 0
                        if print_level >= 2
                            println("    (range) complementarity = ", (λ_temp[i] * (c_x[i] - nlp.meta.ucon[i]), (λ_temp[i] * (c_x[i] - nlp.meta.lcon[i]))), " out of tolerance ϵ = ", ϵ, ". See upper cons " * string(i))

                            if print_level >= 3
                                println("      λ[", i, "]             = ", λ[i])
                                println("      c_x[", i, "]           = ", c_x[i])
                                println("      nlp.meta.ucon[", i, "] = ", nlp.meta.ucon[i])
                                println("      nlp.meta.lcon[", i, "] = ", nlp.meta.lcon[i])
                            end
                        end
                        return false
                    end
                
                end
    
    
    #** III. Lagrangian
        ∇f_x = grad(nlp, x)      
        
        if knitro
            ∇lag_x = ∇f_x - jtprod(nlp, x, λ_temp) - z_L
            #∇lag_x = ∇f_x - jtprod(nlp, x, λ) + z_L

            if norm(∇lag_x, Inf) > ω # Not a stationnary point for the lagrangian
                if print_level >= 2
                    println("    Lagrangian gradient norm = ", norm(∇lag_x, Inf), " is greater than tolerance ω = ", ω)
                    
                    if print_level >= 7
                        println("      ∇f_x              = ", ∇f_x)
                        println("      t(Jac_x)          = ", transpose(jac(nlp, x)))
                        println("      λ_temp            = ", λ_temp)
                        println("      λ                 = ", λ)
                        println("      t(Jac_x) * λ_temp = ", jtprod(nlp, x, λ_temp))
                        println("      t(Jac_x) * λ      = ", jtprod(nlp, x, λ))
                        println("      z                 = ", z)
                    end
                end
                return false
            end
        else
            ∇lag_x = ∇f_x - jtprod(nlp, x, λ_temp) - (z_L - z_U)
            if norm(∇lag_x, Inf) > ω # Not a stationnary point for the lagrangian
                
                if print_level >= 2
                    println("    Lagrangian gradient norm = ", norm(∇lag_x, Inf), " is greater than tolerance ω = ", ω)
                    
                    if print_level >= 7
                        println("      ∇f_x              = ", ∇f_x)
                        println("      t(Jac_x)          = ", transpose(jac(nlp, x)))
                        println("      λ_temp            = ", λ_temp)
                        println("      λ                 = ", λ)
                        println("      t(Jac_x) * λ_temp = ", jtprod(nlp, x, λ_temp))
                        println("      t(Jac_x) * λ      = ", jtprod(nlp, x, λ))
                        println("      - (z_L - z_U)       = ", - (z_L - z_U))
                    end
                end

                if print_level >= 1
                    println("\n  ------- Not fitting with KKT conditions ----------")
                end
                return false
            end
        end
        
        
    
    if print_level >= 1
        println("    " * nlp.meta.name * " solved !")
    end
    return true # all the tests were passed, x, λ respects feasability, complementarity respected, and ∇lag_x(x, λ) almost = 0
end




"""
NCL method implementation. See https://www.researchgate.net/publication/325480151_Stabilized_Optimization_Via_an_NCL_Algorithm for further explications on the method
Arguments: 
    - nlp: optimization problem described by the modelization NLPModels.jl (voir https://github.com/JuliaSmoothOptimizers/NLPModels.jl)
      nlp is the generic problem you want to solve
Returns:
    a GenericExecutionStats, based on the NLPModelsIpopt/Knitro return :
        SolverTools.status                              # of the last resolution #! TODO : Créer un vrai statut
        ncl                                             # the problem in argument,
        solution = sol,                                 # solution found
        iter = k,                                       # number of iteration of the ncl method (not iteration to solve subproblems)
        objective=obj(ncl, sol),                        # objective value
        elapsed_time=0,                                 # time of computation of the whole resolution
        solver_specific=Dict(:multipliers_con => λ_k,   # lagrangian multipliers for : constraints
                            :multipliers_L => z_k_L,    #                              upper bounds
                            :multipliers_U => z_k_U     #                              lower bounds
                            )
        )
"""
function NCLSolve(ncl::NCLModel;                            # Problem to be solved by this method (see NCLModel.jl for further details)
                  tol::Real = 0.001,                        # Tolerance for the gradient lagrangian norm
                  constr_viol_tol::Real = 1e-6,             # Tolerance for the infeasability accepted for ncl
                  compl_inf_tol::Real = 0.0001,             # Tolerance for the complementarity accepted for ncl
                  max_iter_NCL::Int64 = 20,                 # Maximum number of iterations for the NCL method
                  max_iter_solver::Int64 = 1000,            # Maximum number of iterations for the solver (used to solve the subproblem)
                  use_ipopt::Bool = true,                   # Boolean to chose the solver you want to use (true => IPOPT, false => KNITRO)
                  print_level::Int64 = 0,                       # Options for verbosity of printing : 0, nothing; 
                                                                                                     #1, calls to functions and conclusion; 
                                                                                                     #2, calls, little informations on iterations; 
                                                                                                     #3, calls, more information about iterations (and erors in KKT_check); 
                                                                                                     #4, calls, KKT_check, iterations, little information from the solver; 
                                                                                                     #and so on until 14 (increase printing level of the solver)
                  warm_start_init_point = "no"              # "yes" to choose warm start in the subproblem solving. "no" for normal solving.
                ) ::GenericExecutionStats                   # See NLPModelsIpopt / NLPModelsKnitro and SolverTools for further details on this structure

    if print_level >= 1
        println("NCLSolve called on " * ncl.meta.name)
    end
    
    # ** I. Names and variables
        Type = typeof(ncl.meta.x0[1])
        ncl.ρ = 100.0 # step
        ρ_max = 1e10 #biggest penalization authorized

        if warm_start_init_point == "yes"
            mu_init = 1e-3
        else
            mu_init = 0.1
        end

        τ = 10.0 # scale (used to update the ρ_k step)
        α = 0.1 # Constant (α needs to be < 1)
        β = 0.2 # Constant
        #TODO : Pierric

        ω_end = tol #global tolerance, in argument
        ω_k = 1e-8 # sub problem tolerance
        #! change eta_end
        η_end = 1e-6 #constr_viol_tol #global infeasability in argument
        η_k = 1e-2 # sub problem infeasability
        η_min = 1e-8 # smallest infeasability authorized
        ϵ_end = compl_inf_tol #global tolerance for complementarity conditions
        

        # initial points
        x_k = zeros(Type, ncl.nvar_x)
        r_k = zeros(Type, ncl.nvar_r)
        λ_k = zeros(Type, ncl.meta.ncon)
        z_k_U = zeros(Type, length(ncl.meta.uvar))
        z_k_L = zeros(Type, length(ncl.meta.lvar))
        

    # ** II. Optimization loop and return
        k = 0
        converged = false

        while (k <= max_iter_NCL) & !converged
            k += 1
            
            #! TODO choix des mu_init à améliorer...
                if k==2
                    mu_init = 1e-4
                elseif k==4
                    mu_init = 1e-5
                elseif k==6
                    mu_init = 1e-6
                elseif k==8
                    mu_init = 1e-7
                elseif k==10
                    mu_init = 1e-8
                end
            # ** II.1 Get subproblem's solution
                if use_ipopt
                        resolution_k = NLPModelsIpopt.ipopt(ncl, 
                                                            #tol = ω_k, 
                                                            constr_viol_tol = η_k, 
                                                            compl_inf_tol = ϵ_end, 
                                                            print_level = max(print_level - 2, 0), 
                                                            ignore_time = true, 
                                                            warm_start_init_point = warm_start_init_point, 
                                                            mu_init = mu_init, 
                                                            dual_inf_tol=1e-6, 
                                                            max_iter = 3000)
                        
                        # Get variables
                        x_k = resolution_k.solution[1:ncl.nvar_x]
                        r_k = resolution_k.solution[ncl.nvar_x+1 : ncl.nvar_x+ncl.nvar_r]

                        # Get multipliers
                        #! Beware, ipopt doesn't use our convention in KKT_check for constraint multipliers, so we took the opposite. For bound multiplier it seems to work though.
                        λ_k = - resolution_k.solver_specific[:multipliers_con] 
                        z_k_U = resolution_k.solver_specific[:multipliers_U]
                        z_k_L = resolution_k.solver_specific[:multipliers_L]

                else # Knitro
                    resolution_k = _knitro(ncl)::GenericExecutionStats

                    # Get variables
                    x_k = resolution_k.solution.x[1:ncl.nvar_x]
                    r_k = resolution_k.solution.x[ncl.nvar_x+1:ncl.nvar_r]

                    # Get multipliers
                    λ_k = resolution_k.solver_specific[:multipliers_con]
                    z_k_U = resolution_k.solver_specific[:multipliers_U] #! =[] dans ce cas, pas séparé par KNITRO...
                    z_k_L = resolution_k.solver_specific[:multipliers_L]
                end

                if print_level >= 2
                    println("   ----- Iter k = ", k, "-----",
                            "\n    η_k                    = ", η_k,  
                            "\n    norm(r_k,Inf)          = ", norm(r_k,Inf)
                           )
                    
                    if print_level >= 3
                        println("\n      Additionnal information",
                                "\n    ncl.ρ                  = ", ncl.ρ, 
                                "\n    mu_init                = ", mu_init, 
                                # "\n    x_k                    = ", x_k, 
                                "\n    obj(ncl, x_k) = ", obj(ncl.nlp, x_k),
                                "\n    norm(ncl.y - λ_k, Inf) = ", norm(ncl.y - λ_k, Inf),
                               )
                    end
                end

                
                
        # TODO (recherche) : Points intérieurs à chaud...
        # TODO (recherche) : tester la proximité des multiplicateurs λ_k de renvoyés par le solveur et le ncl.y du problème (si r petit, probablement proches.)

            # ** II.2 Treatment & update
                if (norm(r_k,Inf) <= max(η_k, η_end)) | (k == max_iter_NCL) # The residual has decreased enough
                    ncl.y = ncl.y + ncl.ρ * r_k # Updating multiplier
                    η_k = max(η_k/τ, η_min) # η_k / (1 + ncl.ρ ^ β) # (heuristic)
                    if η_k == η_min
                        @warn "min feas = ", η_min, " reached"
                    end
                    #** II.2.1 Solution found ?
                        if (norm(r_k,Inf) <= η_end) | (k == max_iter_NCL) # check if r_k is small enough, or if we've reached the end
                            

                            ## Testing
                            if norm(r_k,Inf) > η_end
                                converged = false
                            else
                                if print_level >= 2
                                    println("    norm(r_k,Inf) = ", norm(r_k,Inf), " <= η_end = ", η_end, " going to KKT_check")
                                end
                #! remove true, put KKT_check
                                converged = true #KKT_check(ncl.nlp, x_k, λ_k, z_k_U[1:ncl.nvar_x], z_k_L[1:ncl.nvar_x], ω_end, η_end, ϵ_end, print_level) 
                            end
                            
                            status = resolution_k.status #TODO: Creer un vrai statut

                            if print_level >= 1
                                if converged
                                    println("    EXIT: optimal solution found")
                                end
                                if k == max_iter_NCL
                                    println("    EXIT: reached max_iter_NCL")
                                end
                            end
                            
                    #** II.2.2 Return if end of the algorithm
                            if converged | (k == max_iter_NCL)
                                return GenericExecutionStats(status, ncl, # TODO optimalité dans le GenericExecutionStats
                                                            solution = resolution_k.solution,
                                                            iter = k,
                                                            objective=obj(ncl, resolution_k.solution), 
                                                            elapsed_time=0,
                                                            solver_specific=Dict(:multipliers_con => λ_k,
                                                                                :multipliers_L => z_k_L,
                                                                                :multipliers_U => z_k_U,
                                                                                :optimal => converged
                                                                                )
                                                            )
                            end
                        end

                else # The residual is to still too large
                    ncl.ρ = τ * ncl.ρ # increase the step # TODO (recherche) : Mieux choisir le pas pour avoir une meilleure convergence
                    #η_k = η_end / (1 + ncl.ρ ^ α) # Change infeasability (heuristic) # ? (simple) η_end ou η_0, cf article
                    if ncl.ρ == ρ_max
                        @warn "max penal = ", ρ_max, " reached"
                    end
                # TODO (recherche...) : update ω_k 
                end
                # ? Chez Nocedal & Wright, p.521, on a : ω_k = 1/ncl.ρ, ncl.ρ = 100ρ_k, η_k = 1/ncl.ρ^0.1
            
        end    
end



printing = false
"""
Main function for the NCL method. 
    Takes an AbstractNLPModel as initial problem, 
    Converts it to a NCLModel,
    Runs NCL method on it, (via the other NCLSolve function)
    Returns (x (solution), y (lagrangian multipliers for constraints), z (lagrangian multpliers for bound constraints))
"""
function NCLSolve(nlp::AbstractNLPModel;                    # Problem to be solved by this method (see NCLModel.jl for further details)
                  use_ipopt::Bool = true,                   # Boolean to chose the solver you want to use (true => IPOPT, false => KNITRO)
                  max_iter_NCL::Int64 = 20,                 # Maximum number of iterations for the NCL method
                  print_level::Int64 = 0,     # Options for printing iterations of the NCL method
                  kwargs...
                ) ::Tuple{GenericExecutionStats, Bool}                   # See NLPModelsIpopt / NLPModelsKnitro and SolverTools for further details on this structure

    #** I. Test : NCL or Ipopt
        if (nlp.meta.ncon == 0) | (nlp.meta.nnln == 0)
            if use_ipopt
                if print_level >= 1
                    println("Résolution de " * nlp.meta.name * " par IPOPT (car 0 résidu ajouté)")
                end
                return (NLPModelsIpopt.ipopt(nlp, print_level = max(print_level-2, 0), kwargs...), true) #tol=tol, constr_viol_tol=constr_viol_tol, compl_inf_tol=compl_inf_tol, max_iter=max_iter_solver, print_level=printing_iterations_solver ? 3 : 0, warm_start_init_point = warm_start_init_point), true)
            else
                if print_level >= 1
                    println("Résolution de " * nlp.meta.name * " par KNITRO (car 0 résidu ajouté)")
                end
                return (_knitro(nlp, max(print_level-2, 0), kwargs...), true) # tol = tol, constr_viol_tol=constr_viol_tol, compl_inf_tol=compl_inf_tol, max_iter = max_iter_solver, print_level = printing_iterations_solver ? 3 : 0, warm_start_init_point = warm_start_init_point), true)
            end

        else

    #** II. NCL Resolution
            ncl = NCLModel(nlp, print_level = print_level)
            if print_level >= 3
                println("\n")
            end

            resol = NCLSolve(ncl, use_ipopt=use_ipopt, max_iter_NCL=max_iter_NCL, print_level=print_level, kwargs...) # tol=tol, constr_viol_tol=constr_viol_tol, compl_inf_tol=compl_inf_tol, max_iter_NCL=max_iter_NCL, max_iter_solver=max_iter_solver, use_ipopt=use_ipopt, printing_iterations=printing_iterations, printing_iterations_solver=printing_iterations_solver, printing_check=printing_check, warm_start_init_point = warm_start_init_point)
            if print_level >= 1
                println("\n")
            end
    
    #** III. Optimality and return
            optimal = !(resol.iter == max_iter_NCL)

            return (resol, optimal)
        end
end