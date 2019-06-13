# comment
#** Important
# ! Warning / Problem
# ? Question
# TODO 

using LinearAlgebra
using NLPModels
using SolverTools
using Ipopt
using NLPModelsIpopt
using CUTEst
using Printf
include("NCLModel.jl")

#using NLPModelsKnitro



######### TODO #########
######### TODO #########
######### TODO #########
    # TODO (feature)   : Créer un vrai statut
    # TODO (feature)   : faire le print dans un fichier externe
    # TODO (feature)   : print de tableau LaTeX aussi

    # TODO (recherche) : choix des mu_init à améliorer...
    # TODO (recherche) : Points intérieurs à chaud...
    # TODO (recherche) : tester la proximité des multiplicateurs λ_k de renvoyés par le solveur et le ncl.y du problème (si r petit, probablement proches.)
    # TODO (recherche) : Mieux choisir le pas pour avoir une meilleure convergence
    # TODO (recherche) : update ω_k 
    # TODO (recherche) : ajuster eta_end

    # TODO (Plus tard) : Pierric pour choix de alpha, beta, tau...
    # TODO (Plus tard) : print de tableau LaTeX aussi












"""
###############################
mult_format_check documentation
    mult_format_check verifys that z_U and z_L are given in the right format to KKT_check.

    !!! Important note !!! The convention is : 
    (P) min f(x)
        s.t. c(x) >= 0

    And then
        multipliers λ >= 0
        Lagrangien(x, λ) = f(x) - λ' * c(x)
        ∇_{x}[lag(x, λ)] = ∇_{x}[f(x)] - t(Jac_{c(x)}) * λ - (z_L - z_U)

###############################
"""
function mult_format_check(z_U::Vector{<:Real}, z_L::Vector{<:Real}, ϵ::Real) ::Tuple{Vector{<:Real}, Vector{<:Real}}
    if (any(z_U .< -ϵ) & any(z_U .> ϵ))
        println("    z_U = ", z_U)

        error("sign problem of z_U passed in argument to KKT_check (detected by mult_format_check function).
               Multipliers are supposed to be >= 0.
               Here, some components are negatives")
    end

    if (any(z_L .< -ϵ) & any(z_L .> ϵ))
        println("    z_L = ", z_L)

        error("sign problem of z_L passed in argument to KKT_check (detected by mult_format_check function). 
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

###############################
###############################













"""
#######################
KKT_check Documentation
    KKT_check tests if (x, λ, z_U, z_L) is a solution of the KKT conditions of the nlp problem (nlp follows the NLPModels.jl formalism, it is suposed to be an AbstractNLPModel), within 
        ω as a tolerance for the lagrangian gradient norm
        η as a tolerance for constraint infeasability
        ϵ as a tolerance for complementarity checking
    the print_level parameter control the verbosity of the function : 0 : nothing
                                                                    # 1 : Function call and result
                                                                    # 2 : Further information in case of failure
                                                                    # 3... : Same, increasing information
                                                                    # 6 & 7 : Shows full vectors, not advised if your problem has a big size

    !!! Important note !!! the lagrangian is considered as :
        l(x, λ) = f(x) - λ' * c(x)          
        with c(x) >= 0
                λ >= 0
    And then
        multipliers λ >= 0
        Lagrangien(x, λ) = f(x) - λ' * c(x)
        ∇_{x}[lag(x, λ)] = ∇_{x}[f(x)] - t(Jac_{c(x)}) * λ - (z_L - z_U)

    Another remark: If z_U is not given (empty), we treat in two different ways complementarity. We can check everything as a range bound constraint in this cae, and when z_L and z_U are given separately, 
#######################
"""
function KKT_check(nlp::AbstractNLPModel,                       # Problem considered
                   x::Vector{<:Real},                           # Potential solution
                   λ::Vector{<:Real},                           # Lagrangian multiplier for constraint
                   z_U::Vector{<:Real},                         # Lagrangian multiplier for upper bound constraint
                   z_L::Vector{<:Real},                         # Lagrangian multiplier for lower bound constraint
                   ω::Real,                                     # Tolerance for lagrangien gradient norm
                   η::Real,                                     # Tolerance or constraint violation
                   ϵ::Real,                                     # Tolerance for complementarity
                   print_level::Int64 ;                         # Verbosity of the function : 0 : nothing
                                                                                            # 1 : Function call and result
                                                                                            # 2 : Further information in case of failure
                                                                                            # 3... : Same, increasing information
                                                                                            # 6 & 7 : Shows full vectors, not advised if your problem has a big size
                   output_file::AbstractString = "KKT.log"      # Path until file for printing details
                  ) ::Bool                                      # true returned if the problem is solved at x, with tolerances specified. false instead.

    

    if print_level >= 1
        file = open(output_file, "w")
        @printf(file, "\nKKT_check called on %s \n", nlp.meta.name)
    end

    #** 0. Warnings et format
        z_U, z_L = mult_format_check(z_U, z_L, ϵ)

        # NLPModelsKnitro returns z_U = [], "didn't find how to treat those separately"
        if (z_U == []) # ? & (nlp.meta.iupp != []) <-  useless ?
            z = z_L
        else
            z = z_L - z_U
        end
        
        if nlp.meta.jfree != []
            error("Problem with free constraints at indices " * string(nlp.meta.jfree) * " passed to KKT_check")
        end
        if nlp.meta.jinf != []
            error("Problem with infeasable constraints at indices " * string(nlp.meta.jinf) * " passed to KKT_check")
        end
        if nlp.meta.iinf != []
            error("Problem with infeasable bound constraints at indices " * string(nlp.meta.iinf) * " passed to KKT_check")
        end
    
    #** I. Bounds constraints
        #** I.1 For free variables
            for i in nlp.meta.ifree 
                #** I.1.1 Feasability USELESS (because variables are free)
                #** I.1.2 Complementarity for bounds 
                    if !(-ϵ <= z[i] <= ϵ)
                        if print_level >= 1
                            if print_level >= 2
                                @printf(file, "    Multiplier not equal to zero for free variable %d \n", i)
                                
                                if print_level >= 3
                                    @printf(file, "      z[%d]             = %7.2e\n", i, z[i])
                                    @printf(file, "      x[%d]             = %7.2e\n", i, x[i])
                                    @printf(file, "      nlp.meta.lvar[%d] = %7.2e\n", i, nlp.meta.lvar[i])
                                    @printf(file, "      nlp.meta.uvar[%d] = %7.2e\n", i, nlp.meta.uvar[i])
                                end
                            end
                            write(file, "\n  ------- Not fitting with KKT conditions ----------\n")
                        end
                        if print_level >= 1
                            close(file)
                        end
                        return false
                    end
            end
        #** I.2 Bounded variables
            for i in vcat(nlp.meta.iupp, nlp.meta.ilow, nlp.meta.ifix, nlp.meta.irng) # bounded = all non free
                #** I.2.1 Feasability
                    if !(nlp.meta.lvar[i] - η <= x[i] <= nlp.meta.uvar[i] + η) 
                        if print_level >= 1
                            if print_level >= 2
                                @printf(file, "    variable %d out of bounds + tolerance\n", i) 
                                
                                if print_level >= 3
                                    @printf(file, "      x[%d] = %7.2e\n", i, x[i])
                                    @printf(file, "      nlp.meta.lvar[%d] = %7.2e\n", i, nlp.meta.lvar[i])
                                    @printf(file, "      nlp.meta.uvar[%d] = %7.2e\n", i, nlp.meta.uvar[i])
                                end
                            end
                            
                            write(file, "\n  ------- Not fitting with KKT conditions ----------\n")
                        end
                        if print_level >= 1
                            close(file)
                        end
                        return false
                    end
                
                #** I.2.2 Complementarity for bounds
                    if !( (-ϵ <= z[i] * (x[i] - nlp.meta.lvar[i]) <= ϵ)  |  (-ϵ <= z[i] * (x[i] - nlp.meta.uvar[i]) <= ϵ) ) # Complementarity condition
                        if print_level >= 1
                            if print_level >= 2
                                @printf(file, "    one of the complementarities = %7.2e or %7.2e is out of tolerance ϵ = %7.2e. See bound var %d\n", z[i] * (x[i] - nlp.meta.lvar[i]), z[i] * (x[i] - nlp.meta.uvar[i]), ϵ, i)

                                if print_level >= 3
                                    @printf(file, "      z[%d]             = %7.2e \n", i, z[i])
                                    @printf(file, "      x[%d]             = %7.2e \n", i, x[i])
                                    @printf(file, "      nlp.meta.lvar[%d] = %7.2e \n", i, nlp.meta.lvar[i])
                                    @printf(file, "      nlp.meta.uvar[%d] = %7.2e \n", i, nlp.meta.uvar[i])
                                end
                            end
                            write(file, "\n  ------- Not fitting with KKT conditions ----------\n")
                        end
                        if print_level >= 1
                            close(file)
                        end
                        return false
                    end
                    
            end
    
    #** II. Other constraints
        #** II.0 Precomputation
            c_x = cons(nlp, x) # Precomputation
            
        #** II.1 Feasability
            for i in 1:nlp.meta.ncon
                if !(nlp.meta.lcon[i] - η <= c_x[i] <= nlp.meta.ucon[i] + η)
                    if print_level >= 1
                        if print_level >= 2
                            @printf(file, "    constraint %d out of bounds + tolerance\n", i)

                            if print_level >= 3
                                @printf(file, "      c_x[%d]               = %7.2e \n", i, c_x[i])
                                @printf(file, "      nlp.meta.ucon[%d] + η = %7.2e \n", i, nlp.meta.ucon[i] + η)
                                @printf(file, "      nlp.meta.lcon[%d] - η = %7.2e \n", i, nlp.meta.lcon[i] - η)
                            end
                        end
                        write(file, "\n  ------- Not fitting with KKT conditions ----------\n")
                    end
                    if print_level >= 1
                        close(file)
                    end
                    return false 
                end
            end

        #** II.2 Complementarity
            for i in 1:nlp.meta.ncon # upper constraints
                if !( (-ϵ <= (λ[i] * (c_x[i] - nlp.meta.ucon[i])) <= ϵ)  |  (-ϵ <= (λ[i] * (c_x[i] - nlp.meta.lcon[i])) <= ϵ) )  # Complementarity condition (for range constraint, we have necessarily : [λ[i] * (c_x[i] - nlp.meta.lcon[i])] * [λ[i] * (c_x[i] - nlp.meta.ucon[i])] = 0
                    if print_level >= 1
                        if print_level >= 2
                            @printf(file, "    one of the two complementarities %7.2e or %7.2e is out of tolerance ϵ = %7.2e. See cons %d \n", i, λ[i] * (c_x[i] - nlp.meta.ucon[i]), (λ[i] * (c_x[i] - nlp.meta.lcon[i])), ϵ)

                            if print_level >= 3
                                @printf(file, "      λ[%d]             = %7.2e \n", i, λ[i])
                                @printf(file, "      c_x[%d]           = %7.2e \n", i, c_x[i])
                                @printf(file, "      nlp.meta.ucon[%d] = %7.2e \n", i, nlp.meta.ucon[i])
                                @printf(file, "      nlp.meta.lcon[%d] = %7.2e \n", i, nlp.meta.lcon[i])
                            end
                        end

                        write(file, "\n  ------- Not fitting with KKT conditions ----------\n")
                    end
                    if print_level >= 1
                        close(file)
                    end
                    return false
                end
            
            end


    #** III. Lagrangian
        #** III.1 Computation
            ∇f_x = grad(nlp, x)      
            if nlp.meta.ncon != 0 # just to avoid DimensionMismatch with ∇f_x - [].
                ∇lag_x = ∇f_x - jtprod(nlp, x, λ) - z
            else
                ∇lag_x = ∇f_x - z
            end
        
        #** III.2 Test, print and return
            if norm(∇lag_x, Inf) > ω # Not a stationnary point for the lagrangian
                if print_level >= 1
                    if print_level >= 2
                        @printf(file, "    Lagrangian gradient norm = %7.2e is greater than tolerance ω = %7.2e \n", norm(∇lag_x, Inf), ω)
                        
                        if 3 <= print_level

                            if print_level >= 7
                                if nlp.meta.ncon != 0
                                    @printf(file, "      ∇f_x         = %7.2e \n", ∇f_x)
                                    @printf(file, "      t(Jac_x)     = %7.2e \n", transpose(jac(nlp, x)))
                                    @printf(file, "      λ            = %7.2e \n", λ)
                                    @printf(file, "      t(Jac_x) * λ = %7.2e \n", jtprod(nlp, x, λ))
                                    @printf(file, "      z_U          = %7.2e \n", z_U)
                                    @printf(file, "      z_L          = %7.2e \n", z_L)
                                    @printf(file, "      ∇lag_x       = %7.2e \n", ∇lag_x)
                                else
                                    @printf(file, "      ∇f_x         = %7.2e \n", ∇f_x)
                                    @printf(file, "      - z          = %7.2e \n", z)
                                    @printf(file, "      ∇lag_x       = %7.2e \n", ∇lag_x)
                                end
                            end
                        end  
                    end
                
                    write(file, "\n  ------- Not fitting with KKT conditions ----------\n")
                end
                if print_level >= 1
                    close(file)
                end
                return false
            end
        
        
    
        if print_level >= 1
            @printf(file, "    %s problem solved !\n", nlp.meta.name)
        end
        
        if print_level >= 1
            close(file)
        end
        return true # all the tests were passed, x, λ respects feasability, complementarity respected, and ∇lag_x(x, λ) almost = 0
end

#######################


















"""
######################
NCLSolve Documentation
    NCL method implementation. See https://www.researchgate.net/publication/325480151_Stabilized_Optimization_Via_an_NCL_Algorithm for further explications on the method
    Arguments: 
        - nlp: optimization problem described by the modelization NLPModels.jl (voir https://github.com/JuliaSmoothOptimizers/NLPModels.jl)
        nlp is the generic problem you want to solve
    Returns:
        a GenericExecutionStats, based on the NLPModelsIpopt/Knitro return :
            SolverTools.status                              # of the last resolution
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
######################
"""
function NCLSolve(ncl::NCLModel ;                           # Problem to be solved by this method (see NCLModel.jl for further details)
                  tol::Real = 0.001,                        # Tolerance for the gradient lagrangian norm
                  constr_viol_tol::Real = 1e-6,             # Tolerance for the infeasability accepted for ncl
                  compl_inf_tol::Real = 0.0001,             # Tolerance for the complementarity accepted for ncl
                  max_iter_NCL::Int64 = 20,                 # Maximum number of iterations for the NCL method
                  max_iter_solver::Int64 = 1000,            # Maximum number of iterations for the solver (used to solve the subproblem)
                  use_ipopt::Bool = true,                   # Boolean to chose the solver you want to use (true => IPOPT, false => KNITRO)
                  output_file::AbstractString = "NCL.log",  # Path to the output file
                  print_level::Int64 = 0,                   # Options for verbosity of printing : 0, nothing; 
                                                                                                # 1, calls to functions and conclusion; 
                                                                                                # 2, calls, little informations on iterations; 
                                                                                                # 3, calls, more information about iterations (and erors in KKT_check); 
                                                                                                # 4, calls, KKT_check, iterations, little information from the solver; 
                                                                                                # and so on until 14 (increase printing level of the solver)
                  warm_start_init_point = "no"              # "yes" to choose warm start in the subproblem solving. "no" for normal solving.
                ) ::GenericExecutionStats                   # See NLPModelsIpopt / NLPModelsKnitro and SolverTools for further details on this structure
    
    
    if print_level >= 1
        file = open(output_file, "w")

        write(file, "NCLSolve called on %s \n", ncl.meta.name)

        if 2 <= print_level <= 5
            @info "If you set print_level above 6, you will get the whole ncl.y, λ_k, x_k, r_k vectors in the NCL iteration print.
                   If you set print_level above 7, you will get the details of the ∇lag_x computation (in case of non fitting KKT conditions)
                   Not advised if your problem has a big size\n"
        end
    end
    
    #** I. Names and variables
        Type = typeof(ncl.meta.x0[1])
        ncl.ρ = 100.0 # step
        ρ_max = 1e15 # biggest penalization authorized
        warm = warm_start_init_point == "yes"

        if warm_start_init_point == "yes"
            mu_init = 1e-3
        else
            mu_init = 0.1
        end

        τ = 10.0 # scale (used to update the ρ_k step)
        α = 0.1 # Constant (α needs to be < 1)
        β = 0.2 # Constant
        

        ω_end = tol #global tolerance, in argument
        ω_k = 1e-8 # sub problem tolerance
        #! change eta_end
        η_end = 1e-6 #constr_viol_tol #global infeasability in argument
        η_k = 1e-2 # sub problem infeasability
        η_min = 1e-10 # smallest infeasability authorized
        ϵ_end = compl_inf_tol #global tolerance for complementarity conditions
        
        if print_level >= 2
            @printf(file, "Optimization parameters")
            @printf(file, "\n    Global tolerance                 ω_end = %7.2e for gradient lagrangian norm", ω_end)
            @printf(file, "\n    Global infeasability             η_end = %7.2e for residuals norm and constraint violation", η_end)
            @printf(file, "\n    Global complementarity tolerance ϵ_end = %7.2e for multipliers and constraints", ϵ_end)
            @printf(file, "\n    Maximal penalization accepted    ρ_max = %7.2e "          , ρ_max)
            @printf(file, "\n    Minimal infeasability accepted   η_min = %7.2e \n", η_min)


            @printf(file, "\n %5s  %4s  %6s", "Iter", "||r_k||_{∞}", "η_k")
            
            if print_level >= 3
                @printf(file, " %6s  %4s  %4s", "ρ", "mu_init", "obj(ncl, x_k)")

                if print_level >= 6
                    @printf(file, " %6s  %5s  %5s \n", "||y||", "||λ_k||", "||x_k||")
                else
                    @printf(file, "\n")
                end

            else
                @printf(file, "\n")
            end
        end


        # initial points
        x_k = zeros(Type, ncl.nvar_x)
        r_k = zeros(Type, ncl.nvar_r)
        norm_r_k_inf = norm(r_k,Inf) #Pre-computation

        λ_k = zeros(Type, ncl.meta.ncon)
        z_k_U = zeros(Type, length(ncl.meta.uvar))
        z_k_L = zeros(Type, length(ncl.meta.lvar))
        

    #** II. Optimization loop and return
        k = 0
        converged = false

        while (k <= max_iter_NCL) & !converged
            #** II.0 Iteration counter and mu_init
                k += 1
                norm_r_k_inf = norm(r_k,Inf) # update

                if (k==2) & warm
                    mu_init = 1e-4
                elseif (k==4) & warm
                    mu_init = 1e-5
                elseif (k==6) & warm
                    mu_init = 1e-6
                elseif (k==8) & warm
                    mu_init = 1e-7
                elseif (k==10) & warm
                    mu_init = 1e-8
                end

            #** II.1 Get subproblem's solution
                #** II.1.2 Solver
                    if use_ipopt
                            resolution_k = NLPModelsIpopt.ipopt(ncl ;
                                                                #tol = ω_k, 
                                                                #constr_viol_tol = η_k, 
                                                                #compl_inf_tol = ϵ_end, 
                                                                print_level = max(print_level - 2, 0), 
                                                                output_file= "ipopt_iter" * string(k),
                                                                ignore_time = true, 
                                                                warm_start_init_point = warm_start_init_point, 
                                                                mu_init = mu_init, 
                                                                dual_inf_tol=1e-6, 
                                                                max_iter = 1000)
                            
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
                
                #** II.1.2 Output print
                    if k % 10 == 0
                        @printf(file, "\n %5s  %4s  %6s", "Iter", "||r_k||_{∞}", "η_k")
                
                        if print_level >= 3
                            @printf(file, " %6s  %4s  %4s", "ρ", "mu_init", "obj(ncl, x_k)")

                            if print_level >= 6
                                @printf(file, " %6s  %5s  %5s \n", "||y||", "||λ_k||", "||x_k||")
                            else
                                @printf(file, "\n")
                            end

                        else
                            @printf(file, "\n")
                        end
                    end

                    if print_level >= 2 
                        @printf(file, "\n %2d  %7.1e  %7.1e", k, norm_r_k_inf, η_k)
                
                        if print_level >= 3
                            @printf(file, " %7.1e  %7.1e  %7.1e", ncl.ρ, mu_init, obj(ncl, x_k))

                            if print_level >= 6
                                @printf(file, " %7.1e  %7.1e  %7.1e \n", norm(ncl.y, Inf), norm(λ_k, Inf), norm(x_k))
                            else
                                @printf(file, "\n")
                            end
                            
                        else
                            @printf(file, "\n")
                        end
                    end


            #** II.2 Treatment & update
                if (norm_r_k_inf <= max(η_k, η_end)) | (k == max_iter_NCL) # The residual has decreased enough
                    #** II.2.1 Update
                        ncl.y = ncl.y + ncl.ρ * r_k # Updating multiplier
                        η_k = max(η_k/τ, η_min) # η_k / (1 + ncl.ρ ^ β) # (heuristic)

                        if η_k == η_min
                            @warn "Minimum constraint violation η_min = " * string(η_min) * " reached"
                        end
                    
                    #** II.2.2 Solution found ?
                        if (norm_r_k_inf <= η_end) | (k == max_iter_NCL) # check if r_k is small enough, or if we've reached the end
                            ## Testing
                            if norm_r_k_inf > η_end
                                converged = false
                            else
                                if print_level >= 2
                                    @printf(file, "--------\n   norm(r_k,Inf) = %7.2e  <= η_end = %7.2e. Calling KKT_check\n", norm_r_k_inf, η_end)
                                end

                        #! mettre converged = true pour tester avec AMPL
                                converged = KKT_check(ncl.nlp, x_k, λ_k, z_k_U[1:ncl.nvar_x], z_k_L[1:ncl.nvar_x], ω_end, η_end, ϵ_end, print_level, output_file = output_file)
                            end
                            
                            status = resolution_k.status

                            if print_level >= 1
                                if converged
                                    write(file, "---------------\n",
                                                "   EXIT: optimal solution found\n")
                                end
                                if k == max_iter_NCL
                                    write(file, "----------------\n",
                                                "   EXIT: reached max_iter_NCL\n")
                                end
                            end
                            
                    #** II.2.3 Return if end of the algorithm
                            if converged | (k == max_iter_NCL)
                                if print_level >= 1
                                    close(file)
                                end
                                return GenericExecutionStats(status, ncl, 
                                                            solution = resolution_k.solution,
                                                            iter = k,
                                                            objective=obj(ncl, resolution_k.solution), 
                                                            elapsed_time=0,
                                                            solver_specific=Dict(:multipliers_con => λ_k,
                                                                                :multipliers_L => z_k_L,
                                                                                :multipliers_U => z_k_U,
                                                                                :internal_msg => converged ? Symbol("Solve_Succeeded") : Symbol("Solve_Failed")
                                                                                )
                                                            )
                            end
                        end

                else 
            #** II.3 Increase penalization
                    ncl.ρ = τ * ncl.ρ # increase the step 
                    #η_k = η_end / (1 + ncl.ρ ^ α) # Change infeasability (heuristic)
                    if ncl.ρ == ρ_max
                        @warn "Maximum penalization ρ = " * string(ρ_max) * " reached"
                    end
                end
                # ? Chez Nocedal & Wright, p.521, on a : ω_k = 1/ncl.ρ, ncl.ρ = 100ρ_k, η_k = 1/ncl.ρ^0.1
            
        end    
end

######################
######################

















"""
#################################
Main function for the NCL method. 
    Takes an AbstractNLPModel as initial problem, 
    Converts it to a NCLModel,
    Runs NCL method on it, (via the other NCLSolve function)
    Returns (x (solution), y (lagrangian multipliers for constraints), z (lagrangian multpliers for bound constraints))
#################################
"""
function NCLSolve(nlp::AbstractNLPModel;                    # Problem to be solved by this method (see NCLModel.jl for further details)
                  use_ipopt::Bool = true,                   # Boolean to chose the solver you want to use (true => IPOPT, false => KNITRO)
                  max_iter_NCL::Int64 = 20,                 # Maximum number of iterations for the NCL method
                  linear_residuals::Bool = true,            # Boolean to choose if you want residuals onlinear constraints (true), or not (false)
                  print_level::Int64 = 0,                   # Options for printing iterations of the NCL method
                  output_file::AbstractString = "solver.log",
                  kwargs...                                 # Other arguments for the other NCLSolve function
                ) ::GenericExecutionStats                   # See NLPModelsIpopt / NLPModelsKnitro and SolverTools for further details on this structure

    #** I. Test : NCL or Ipopt
        if (nlp.meta.ncon == 0) | (nlp.meta.nnln == 0)
            if use_ipopt
                if print_level >= 1
                    println("Résolution de " * nlp.meta.name * " par IPOPT (car 0 résidu ajouté)")
                end
                
                return NLPModelsIpopt.ipopt(nlp, print_level = max(print_level-2, 0), output_file=output_file, kwargs...) #tol=tol, constr_viol_tol=constr_viol_tol, compl_inf_tol=compl_inf_tol, max_iter=max_iter_solver, print_level=printing_iterations_solver ? 3 : 0, warm_start_init_point = warm_start_init_point), true)
            else
                if print_level >= 1
                    println("Résolution de " * nlp.meta.name * " par KNITRO (car 0 résidu ajouté)")
                end
                
                return _knitro(nlp, max(print_level-2, 0), kwargs...) # tol = tol, constr_viol_tol=constr_viol_tol, compl_inf_tol=compl_inf_tol, max_iter = max_iter_solver, print_level = printing_iterations_solver ? 3 : 0, warm_start_init_point = warm_start_init_point), true)
            end

        else

    #** II. NCL Resolution
        ncl = NCLModel(nlp ; print_level = print_level, res_lin_cons = linear_residuals)
        resol = NCLSolve(ncl ; use_ipopt=use_ipopt, max_iter_NCL=max_iter_NCL, print_level=print_level, kwargs...) # tol=tol, constr_viol_tol=constr_viol_tol, compl_inf_tol=compl_inf_tol, max_iter_NCL=max_iter_NCL, max_iter_solver=max_iter_solver, use_ipopt=use_ipopt, printing_iterations=printing_iterations, printing_iterations_solver=printing_iterations_solver, printing_check=printing_check, warm_start_init_point = warm_start_init_point)
    
    #** III. Optimality and return
            if print_level >= 2
                println("   EXIT global NCLSolve function : ", resol.solver_specific[:internal_msg], "\n")
            end

            return resol
        end
end
#################################


































#####################
## TESTS FUNCTIONS ##

    """
    #############################
    Unitary tests for NLCModel.jl
    #############################
    """
    function test_NLCModel(test::Bool) ::Test.DefaultTestSet
        # Test parameters
            ρ = 1.
            y = [2., 1.]
            g = Vector{Float64}(undef,4)
            cx = Vector{Float64}(undef,4)
            
            hrows = [1, 2, 2, 3, 4]
            hcols = [1, 1, 2, 3, 4]
            hvals = Vector{Float64}(undef,5)
            Hv = Vector{Float64}(undef,4)

            jrows = [1, 2, 3, 4, 1, 2, 3, 4, 3, 4]
            jcols = [1, 1, 1, 1, 2, 2, 2, 2, 3, 4]
            jvals = Vector{Float64}(undef,10)
            Jv = Vector{Float64}(undef,4)
            
        # Test problem
            f(x) = x[1] + x[2]
            x0 = [0.5, 0.5]
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

            name = "Unitary test problem"
            nlp = ADNLPModel(f, x0 ; lvar=lvar, uvar=uvar, c=c, lcon=lcon, ucon=ucon, name=name, lin = [1,3])::ADNLPModel
            nlc_nlin_res = NCLModel(nlp ; res_lin_cons = false)::NCLModel

            nlc_nlin_res.y = y
            nlc_nlin_res.ρ = ρ


            nlc_cons_res = NCLModel(nlp, res_lin_cons = true)::NCLModel
            nlc_cons_res.ρ = ρ

        # Unitary tests
            if test
                @testset "NCLModel. No linear residuals" begin
                    @testset "NCLModel struct" begin
                        @testset "NCLModel struct information about nlp" begin
                            @test nlc_nlin_res.nvar_x == 2 
                            @test nlc_nlin_res.nvar_r == 2 # two non linear constraint, so two residues
                            @test nlc_nlin_res.minimize == true
                            @test nlc_nlin_res.jres == [2,4]
                        end

                        @testset "NCLModel struct constant parameters" begin
                            @test nlc_nlin_res.nvar == 4 # 2 x, 2 r
                            @test nlc_nlin_res.meta.lvar == [0., 0., -Inf, -Inf] # no bounds for residues
                            @test nlc_nlin_res.meta.uvar == [1., 1., Inf, Inf]
                            @test nlc_nlin_res.meta.x0 == [0.5, 0.5, 1., 1.]
                            @test nlc_nlin_res.meta.y0 == [0., 0., 0., 0.]
                            @test nlc_nlin_res.y == y
                            @test length(nlc_nlin_res.y) == nlc_nlin_res.nvar_r
                            @test nlc_nlin_res.meta.nnzj == nlp.meta.nnzj + 2 # 2 residues, one for each non linear constraint
                            @test nlc_nlin_res.meta.nnzh == nlp.meta.nnzh + 2 # add a digonal of ρ
                        end
                    end

                    @testset "NCLModel f" begin
                        @test obj(nlc_nlin_res, [0., 0., 0., 0.]) == 0.
                        @test obj(nlc_nlin_res, [0.5, 0.5, 0., -1.]) == 1. - 1. + 0.5 * ρ * 1.
                    end

                    @testset "NCLModel ∇f" begin 
                        @testset "NCLModel grad()" begin
                            @test grad(nlc_nlin_res, [0., 0., 0., 0.]) == [1., 1., 2., 1.]
                            @test grad(nlc_nlin_res, [0.5, 0.5, 0., -1.]) == [1., 1., 2., 1. - ρ]
                        end

                        @testset "NCLModel grad!()" begin
                            @test grad!(nlc_nlin_res, [0., 0., 0., 0.], g) == [1., 1., 2., 1.]
                            @test grad!(nlc_nlin_res, [0.5, 0.5, 0., -1.], zeros(4)) == [1., 1., 2., 1. - ρ]
                        end
                    end

                    @testset "NCLModel hessian of the lagrangian" begin
                        @testset "NCLModel hessian of the lagrangian hess()" begin
                            @test hess(nlc_nlin_res, [0., 0., 0., 0.], y=zeros(Float64,4)) == [0. 0. 0. 0. ; 
                                                                                    0. 0. 0. 0. ;
                                                                                    0. 0. ρ  0. ;
                                                                                    0. 0. 0. ρ]
                            @test hess(nlc_nlin_res, nlc_nlin_res.meta.x0, y=[1.,1.,1.,1.]) == [2. 0. 0. 0. ; #not symetrical because only the lower triangle is returned by hess
                                                                            1. 0. 0. 0. ;
                                                                            0. 0. ρ  0. ;
                                                                            0. 0. 0. ρ]
                        end               

                        @testset "NCLModel hessian of the lagrangian hess_coord()" begin
                            @test hess_coord(nlc_nlin_res, [0., 0., 0., 0.], y = zeros(Float64,4))[1][nlp.meta.nnzh+1 : nlp.meta.nnzh+2] == [3, 4]
                            @test hess_coord(nlc_nlin_res, nlc_nlin_res.meta.x0, y = [1.,1.,1.,1.])[1][nlp.meta.nnzh+1 : nlp.meta.nnzh+2] == [3, 4]

                            @test hess_coord(nlc_nlin_res, [0., 0., 0., 0.], y = zeros(Float64,4))[2][nlp.meta.nnzh+1 : nlp.meta.nnzh+2] == [3, 4]
                            @test hess_coord(nlc_nlin_res, nlc_nlin_res.meta.x0, y = [1.,1.,1.,1.])[2][nlp.meta.nnzh+1 : nlp.meta.nnzh+2] == [3, 4]

                            @test hess_coord(nlc_nlin_res, [0., 0., 0., 0.], y = zeros(Float64,4))[3][nlp.meta.nnzh+1 : nlp.meta.nnzh+2] == [ρ, ρ]
                            @test hess_coord(nlc_nlin_res, nlc_nlin_res.meta.x0, y = [1.,1.,1.,1.])[3][nlp.meta.nnzh+1 : nlp.meta.nnzh+2] == [ρ, ρ]
                        end

                        @testset "NCLModel hessian of the lagrangian hess_coord!()" begin
                            @test hess_coord!(nlc_nlin_res, [0., 0., 0., 0.], hrows, hcols, hvals, y = zeros(Float64,4))[1] == hrows
                            @test hess_coord!(nlc_nlin_res, nlc_nlin_res.meta.x0, hrows, hcols, hvals, y = [1.,1.,1.,1.])[1] == hrows

                            @test hess_coord!(nlc_nlin_res, [0., 0., 0., 0.], hrows, hcols, hvals, y = zeros(Float64,4))[2] == hcols
                            @test hess_coord!(nlc_nlin_res, nlc_nlin_res.meta.x0, hrows, hcols, hvals, y = [1.,1.,1.,1.])[2] == hcols

                            @test hess_coord!(nlc_nlin_res, [0., 0., 0., 0.], hrows, hcols, hvals, y = zeros(Float64,4))[3][nlp.meta.nnzh+1 : nlp.meta.nnzh+2] == [ρ, ρ]
                            @test hess_coord!(nlc_nlin_res, nlc_nlin_res.meta.x0, hrows, hcols, hvals, y = [1.,1.,1.,1.])[3][nlp.meta.nnzh+1 : nlp.meta.nnzh+2] == [ρ, ρ]
                        end


                        @testset "NCLModel hessian of the lagrangian hess_structure()" begin
                            @test hess_structure(nlc_nlin_res)[1] == vcat(hess_structure(nlc_nlin_res.nlp)[1], [3, 4])
                            @test hess_structure(nlc_nlin_res)[2] == vcat(hess_structure(nlc_nlin_res.nlp)[2], [3, 4])
                        end

                        @testset "NCLModel hessian of the lagrangian hprod()" begin
                            @test hprod(nlc_nlin_res, nlc_nlin_res.meta.x0, [1,2,3,4], y = [1.,1.,1.,1.]) == [4,1,3*ρ,4*ρ]
                        end

                        @testset "NCLModel hessian of the lagrangian hprod!()" begin
                            @test hprod!(nlc_nlin_res, nlc_nlin_res.meta.x0, [1,2,3,4], y = [1.,1.,1.,1.], Hv) == [4,1,3*ρ,4*ρ]
                        end
                    end

                    @testset "NCLModel constraint" begin
                        @testset "NCLModel constraint cons()" begin
                            @test size(cons(nlc_nlin_res, [1.,1.,0.,1.]), 1) == 4
                            @test cons(nlc_nlin_res, [1.,1.,0.,1.]) == [0.,2.,0.,2.]
                            @test cons(nlc_nlin_res, [1.,0.5,1.,1.]) == [0.5,2.5,0.5,1.5]
                        end
                        @testset "NCLModel constraint cons!()" begin
                            @test size(cons!(nlc_nlin_res, [1.,1.,0.,1.], cx), 1) == 4
                            @test cons!(nlc_nlin_res, [1.,1.,0.,1.], cx) == [0.,2.,0.,2.]
                            @test cons!(nlc_nlin_res, [1.,0.5,1.,1.], cx) == [0.5,2.5,0.5,1.5]
                        end
                    end

                    @testset "NCLModel constraint jacobian" begin
                        @testset "NCLModel constraint jac()" begin
                            @test jac(nlc_nlin_res, [1.,1.,0.,1.]) == [1 -1 0 0 ;
                                                            2  1 1 0 ;
                                                            1 -1 0 0 ;
                                                            1  1 0 1 ]

                            @test jac(nlc_nlin_res, [1.,0.5,1.,1.]) == [ 1 -1  0  0 ;
                                                                2  1  1  0 ;
                                                                1 -1  0  0 ;
                                                                0.5 1  0  1]
                        end
                        
                        @testset "NCLModel constraint jac_coord()" begin
                            @test jac_coord(nlc_nlin_res, [1.,1.,0.,1.])[1][9:10] == [2,4]
                            @test jac_coord(nlc_nlin_res, [1.,1.,0.,1.])[2][9:10] == [3,4]
                            @test jac_coord(nlc_nlin_res, [1.,0.5,1.,1.])[3][9:10] == [1,1]
                        end

                        @testset "NCLModel constraint jac_coord!()" begin
                            @test jac_coord!(nlc_nlin_res, [1.,1.,0.,1.], jrows, jcols, jvals)[1] == jrows
                            @test jac_coord!(nlc_nlin_res, [1.,1.,0.,1.], jrows, jcols, jvals)[2] == jcols
                            @test jac_coord!(nlc_nlin_res, [1.,1.,0.,1.], jrows, jcols, jvals)[3] == [1,2,1,1,-1,1,-1,1,1,1]
                            @test jac_coord!(nlc_nlin_res, [1.,0.5,1.,1.], jrows, jcols, jvals)[3] == [1,2,1,0.5,-1,1,-1,1,1,1]
                        end

                        @testset "NCLModel constraint jac_struct()" begin
                            @test jac_structure(nlc_nlin_res)[1][9:10] == [2,4]
                            @test jac_structure(nlc_nlin_res)[2][9:10] == [3,4]
                        end

                        @testset "NCLModel constraint jprod()" begin
                            @test jprod(nlc_nlin_res, [1.,1.,0.,1.], [1.,1.,1.,1.]) == [0,4,0,3]
                            @test jprod(nlc_nlin_res, [1.,0.5,1.,1.], [0.,1.,0.,1.]) == [-1,1,-1,2]
                        end

                        @testset "NCLModel constraint jprod!()" begin
                            @test jprod!(nlc_nlin_res, [1.,1.,0.,1.], [1.,1.,1.,1.], Jv) == [0,4,0,3]
                            @test jprod!(nlc_nlin_res, [1.,0.5,1.,1.], [0.,1.,0.,1.], Jv) == [-1,1,-1,2]
                        end

                        @testset "NCLModel constraint jtprod()" begin
                            @test jtprod(nlc_nlin_res, [1.,1.,0.,1.], [1.,1.,1.,1.]) == [5,0,1,1]
                            @test jtprod(nlc_nlin_res, [1.,0.5,1.,1.], [0.,1.,0.,1.]) == [2.5,2,1,1]
                        end

                        @testset "NCLModel constraint jtprod!()" begin
                            @test jtprod!(nlc_nlin_res, [1.,1.,0.,1.], [1.,1.,1.,1.], Jv) == [5,0,1,1]
                            @test jtprod!(nlc_nlin_res, [1.,0.5,1.,1.], [0.,1.,0.,1.], Jv) == [2.5,2,1,1]
                        end
                    end
                end

                @testset "NCLModel. All residuals" begin
                    @testset "NCLModel struct" begin
                        @testset "NCLModel struct information about nlp" begin
                            @test nlc_cons_res.nvar_x == 2 
                            @test nlc_cons_res.nvar_r == 4 # two non linear constraint, so two residues
                            @test nlc_cons_res.minimize == true
                            @test nlc_cons_res.jres == []
                        end

                        @testset "NCLModel struct constant parameters" begin
                            @test nlc_cons_res.nvar == 6 # 2 x, 4 r
                            @test nlc_cons_res.meta.lvar == [0., 0., -Inf, -Inf, -Inf, -Inf] # no bounds for residues
                            @test nlc_cons_res.meta.uvar == [1., 1., Inf, Inf, Inf, Inf]
                            @test nlc_cons_res.meta.x0 == [0.5, 0.5, 1., 1., 1., 1.]
                            @test nlc_cons_res.meta.y0 == [0., 0., 0., 0.]
                            @test nlc_cons_res.y == [1., 1., 1., 1.]
                            @test length(nlc_cons_res.y) == nlc_cons_res.nvar_r
                            @test nlc_cons_res.meta.nnzj == nlp.meta.nnzj + 4 # 2 residues, one for each constraint
                            @test nlc_cons_res.meta.nnzh == nlp.meta.nnzh + 4 # add a digonal of ρ
                        end
                    end

                    @testset "NCLModel f" begin
                        @test obj(nlc_cons_res, [0., 0., 0., 0., 0., 0.]) == 0.
                        @test obj(nlc_cons_res, [0.5, 0.5, 0., -1., 0., 1.]) == 1. + 0. + 0.5 * ρ * (1. + 1.)
                    end

                    @testset "NCLModel ∇f" begin 
                        @testset "NCLModel grad()" begin
                            @test grad(nlc_cons_res, [0., 0., 0., 0., 0., 0.]) == [1., 1., 1., 1., 1., 1.]
                            @test grad(nlc_cons_res, [0.5, 0.5, 0., -1., 0., 1.]) == [1., 1., 1., 1. - ρ, 1., 1 + ρ]
                        end

                        @testset "NCLModel grad!()" begin
                            @test grad!(nlc_cons_res, [0., 0., 0., 0., 0., 0.], vcat(g, [1,2])) == [1., 1., 1., 1., 1., 1.]
                            @test grad!(nlc_cons_res, [0.5, 0.5, 0., -1., 0., 1.], zeros(6)) == [1., 1., 1., 1. - ρ, 1., 1 + ρ]
                        end
                    end

                    @testset "NCLModel hessian of the lagrangian" begin
                        @testset "NCLModel hessian of the lagrangian hess()" begin
                            @test hess(nlc_cons_res, [0., 0., 0., 0.], y=zeros(Float64,6)) == [0. 0. 0. 0. 0. 0. ; 
                                                                                            0. 0. 0. 0. 0. 0. ;
                                                                                            0. 0. ρ  0. 0. 0. ;
                                                                                            0. 0. 0. ρ  0. 0. ;
                                                                                            0. 0. 0. 0. ρ  0. ;
                                                                                            0. 0. 0. 0. 0. ρ ]
                            @test hess(nlc_cons_res, nlc_cons_res.meta.x0, y=[1.,1.,1.,1.]) == [2. 0. 0. 0. 0. 0. ; #not symetrical because only the lower triangle is returned by hess
                                                                                                1. 0. 0. 0. 0. 0. ;
                                                                                                0. 0. ρ  0. 0. 0. ;
                                                                                                0. 0. 0. ρ  0. 0. ;
                                                                                                0. 0. 0. 0. ρ  0. ;
                                                                                                0. 0. 0. 0. 0. ρ ]
                        end               

                        @testset "NCLModel hessian of the lagrangian hess_coord()" begin
                            @test hess_coord(nlc_cons_res, [0., 0., 0., 0., 0., 0.], y = zeros(Float64,6))[1][nlp.meta.nnzh+1 : nlp.meta.nnzh+nlc_cons_res.nvar_r] == [3, 4, 5, 6]
                            @test hess_coord(nlc_cons_res, nlc_cons_res.meta.x0, y = [1.,1.,1.,1.,1.,1.])[1][nlp.meta.nnzh+1 : nlp.meta.nnzh+nlc_cons_res.nvar_r] == [3, 4, 5, 6]

                            @test hess_coord(nlc_cons_res, [0., 0., 0., 0., 0., 0.], y = zeros(Float64,6))[2][nlp.meta.nnzh+1 : nlp.meta.nnzh+nlc_cons_res.nvar_r] == [3, 4, 5, 6]
                            @test hess_coord(nlc_cons_res, nlc_cons_res.meta.x0, y = [1.,1.,1.,1.,1.,1.])[2][nlp.meta.nnzh+1 : nlp.meta.nnzh+nlc_cons_res.nvar_r] == [3, 4, 5, 6]

                            @test hess_coord(nlc_cons_res, [0., 0., 0., 0., 0., 0.], y = zeros(Float64,6))[3][nlp.meta.nnzh+1 : nlp.meta.nnzh+nlc_cons_res.nvar_r] == [ρ, ρ, ρ, ρ]
                            @test hess_coord(nlc_cons_res, nlc_cons_res.meta.x0, y = [1.,1.,1.,1.,1.,1.])[3][nlp.meta.nnzh+1 : nlp.meta.nnzh+nlc_cons_res.nvar_r] == [ρ, ρ, ρ, ρ]
                        end

                        @testset "NCLModel hessian of the lagrangian hess_coord!()" begin
                            @test hess_coord!(nlc_cons_res, [0., 0., 0., 0., 0., 0.], vcat(hrows, [5, 6]), vcat(hcols, [5, 6]), vcat(hvals, [5, 6]), y = zeros(Float64,6))[1] == vcat(hrows, [5, 6])
                            @test hess_coord!(nlc_cons_res, nlc_cons_res.meta.x0, vcat(hrows, [5, 6]), vcat(hcols, [5, 6]), vcat(hvals, [5, 6]), y = [1.,1.,1.,1.,1.,1.])[1] == vcat(hrows, [5, 6])

                            @test hess_coord!(nlc_cons_res, [0., 0., 0., 0., 0., 0.], vcat(hrows, [5, 6]), vcat(hcols, [5, 6]), vcat(hvals, [5, 6]), y = zeros(Float64,6))[2] == vcat(hcols, [5, 6])
                            @test hess_coord!(nlc_cons_res, nlc_cons_res.meta.x0, vcat(hrows, [5, 6]), vcat(hcols, [5, 6]), vcat(hvals, [5, 6]), y = [1.,1.,1.,1.,1.,1.])[2] == vcat(hcols, [5, 6])

                            @test hess_coord!(nlc_cons_res, [0., 0., 0., 0., 0., 0.], vcat(hrows, [5, 6]), vcat(hcols, [5, 6]), vcat(hvals, [5, 6]), y = zeros(Float64,6))[3][nlp.meta.nnzh+1 : nlp.meta.nnzh+nlc_cons_res.nvar_r] == [ρ, ρ, ρ, ρ]
                            @test hess_coord!(nlc_cons_res, nlc_cons_res.meta.x0, vcat(hrows, [5, 6]), vcat(hcols, [5, 6]), vcat(hvals, [5, 6]), y = [1.,1.,1.,1.,1.,1.])[3][nlp.meta.nnzh+1 : nlp.meta.nnzh+nlc_cons_res.nvar_r] == [ρ, ρ, ρ, ρ]
                        end


                        @testset "NCLModel hessian of the lagrangian hess_structure()" begin
                            @test hess_structure(nlc_cons_res)[1] == vcat(hess_structure(nlc_cons_res.nlp)[1], [3, 4, 5, 6])
                            @test hess_structure(nlc_cons_res)[2] == vcat(hess_structure(nlc_cons_res.nlp)[2], [3, 4, 5, 6])
                        end

                        @testset "NCLModel hessian of the lagrangian hprod()" begin
                            @test hprod(nlc_cons_res, nlc_cons_res.meta.x0, [1,2,3,4,5,6], y = [1.,1.,1.,1.,1.,1.]) == [4,1,3*ρ,4*ρ,5*ρ,6*ρ]
                        end

                        @testset "NCLModel hessian of the lagrangian hprod!()" begin
                            @test hprod!(nlc_cons_res, nlc_cons_res.meta.x0, [1,2,3,4,5,6], y = [1.,1.,1.,1.,1.,1.], vcat(Hv, [0.,0.])) == [4,1,3*ρ,4*ρ,5*ρ,6*ρ]
                        end
                    end

                    @testset "NCLModel constraint" begin
                        @testset "NCLModel constraint cons()" begin
                            @test size(cons(nlc_cons_res, [1.,1.,0.,1.,1.,1.]), 1) == 4
                            @test cons(nlc_cons_res, [1.,1.,0.,1.,1.,1.]) == [0.,3.,1.,2.]
                            @test cons(nlc_cons_res, [1.,0.5,1.,1.,0.,-1.]) == [1.5,2.5,0.5,-0.5]
                        end
                        @testset "NCLModel constraint cons!()" begin
                            @test size(cons!(nlc_cons_res, [1.,1.,0.,1.,1.,1.], cx), 1) == 4
                            @test cons!(nlc_cons_res, [1.,1.,0.,1.,1.,1.], cx) == [0.,3.,1.,2.]
                            @test cons!(nlc_cons_res, [1.,0.5,1.,1.,0.,-1.], cx) == [1.5,2.5,0.5,-0.5]
                        end
                    end

                    @testset "NCLModel constraint jacobian" begin
                        @testset "NCLModel constraint jac()" begin
                            @test jac(nlc_cons_res, [1.,1.,0.,1.,1.,1.]) == [1 -1  1  0  0  0;
                                                                            2  1  0  1  0  0;
                                                                            1 -1  0  0  1  0;
                                                                            1  1  0  0  0  1]

                            @test jac(nlc_cons_res, [1.,0.5,1.,1.,0.,-1.]) == [1  -1  1  0  0  0;
                                                                            2   1  0  1  0  0;
                                                                            1  -1  0  0  1  0;
                                                                            0.5  1  0  0  0  1]
                        end
                        
                        @testset "NCLModel constraint jac_coord()" begin
                            @test jac_coord(nlc_cons_res, [1.,1.,0.,1.,1.,1.])[1][9:12] == [1,2,3,4]
                            @test jac_coord(nlc_cons_res, [1.,1.,0.,1.,1.,1.])[2][9:12] == [3,4,5,6]
                            @test jac_coord(nlc_cons_res, [1.,0.5,1.,1.,0.,-1.])[3][9:12] == [1,1,1,1]
                        end

                        @testset "NCLModel constraint jac_coord!()" begin
                            @test jac_coord!(nlc_cons_res, [1.,1.,0.,1.,1.,1.], vcat(jrows, [1,2]), vcat(jcols, [0,0]), vcat(jvals, [1,21]))[1] == vcat(jrows, [1,2])
                            @test jac_coord!(nlc_cons_res, [1.,1.,0.,1.,1.,1.], vcat(jrows, [1,2]), vcat(jcols, [0,0]), vcat(jvals, [1,21]))[2] == vcat(jcols, [0,0])
                            @test jac_coord!(nlc_cons_res, [1.,1.,0.,1.,1.,1.], vcat(jrows, [1,2]), vcat(jcols, [0,0]), vcat(jvals, [1,21]))[3] == [1,2,1,1,-1,1,-1,1,1,1,1,1]
                            @test jac_coord!(nlc_cons_res, [1.,0.5,1.,1.,0.,-1.], vcat(jrows, [1,2]), vcat(jcols, [0,0]), vcat(jvals, [1,21]))[3] == [1,2,1,0.5,-1,1,-1,1,1,1,1,1]
                        end

                        @testset "NCLModel constraint jac_struct()" begin
                            @test jac_structure(nlc_cons_res)[1][9:12] == [1,2,3,4]
                            @test jac_structure(nlc_cons_res)[2][9:12] == [3,4,5,6]
                        end

                        @testset "NCLModel constraint jprod()" begin
                            @test jprod(nlc_cons_res, [1.,1.,0.,1.,1.,1.], [1.,1.,1.,1.,1.,1.]) == [1,4,1,3]
                            @test jprod(nlc_cons_res, [1.,0.5,1.,1.,0.,-1.], [0.,1.,0.,1.,-1.,-1.]) == [-1,2,-2,0]
                        end

                        @testset "NCLModel constraint jprod!()" begin
                            @test jprod!(nlc_cons_res, [1.,1.,0.,1.,1.,1.], [1.,1.,1.,1.,1.,1.], Jv) == [1,4,1,3]
                            @test jprod!(nlc_cons_res, [1.,0.5,1.,1.,0.,-1.], [0.,1.,0.,1.,-1.,-1.], Jv) == [-1,2,-2,0]
                        end

                        @testset "NCLModel constraint jtprod()" begin
                            @test jtprod(nlc_cons_res, [1.,1.,0.,1.,1.,1.], [1.,1.,1.,1.]) == [5,0,1,1,1,1]
                            @test jtprod(nlc_cons_res, [1.,0.5,1.,1.,0.,-1.], [0.,1.,0.,1.]) == [2.5,2,0,1,0,1]
                        end

                        @testset "NCLModel constraint jtprod!()" begin
                            @test jtprod!(nlc_cons_res, [1.,1.,0.,1.,1.,1.], [1.,1.,1.,1.], vcat(Jv, [0,1])) == [5,0,1,1,1,1]
                            @test jtprod!(nlc_cons_res, [1.,0.5,1.,1.,0.,-1.], [0.,1.,0.,1.], vcat(Jv, [0,1])) == [2.5,2,0,1,0,1]
                        end
                    end
                end
            else
                @testset "Empty test" begin
                    @test true
                end
            end
    end
    #############################
    #############################




    """
    #############################
    Unitary tests for NCLSolve.jl
    #############################
    """
    function test_NCLSolve(test::Bool ; HS_begin_KKT::Int64 = 13, HS_end_KKT::Int64 = 13, HS_begin_NCL::Int64 = 25,  HS_end_NCL::Int64 = 25) ::Test.DefaultTestSet
        # Test parameters
            print_level = 0
            ω = 0.001
            η = 0.0001
            ϵ = 0.0001
            probs_KKT = ["HS" * string(i) for i in HS_begin_KKT:HS_end_KKT]
            probs_NCL = ["HS" * string(i) for i in HS_begin_NCL:HS_end_NCL]

        # Test problem
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

            name = "Unitary test problem"
            nlp = ADNLPModel(f, x0; lvar=lvar, uvar=uvar, c=c, lcon=lcon, ucon=ucon, name=name, lin = [1,3])::ADNLPModel
            ncl_nlin_res = NCLModel(nlp, res_lin_cons = false)::NCLModel

            ncl_nlin_res.y = y
            ncl_nlin_res.ρ = ρ

            nlc_cons_res = NCLModel(nlp, res_lin_cons = true)::NCLModel

        if test
            @testset "KKT_check function" begin
                for name in probs_KKT # several tests
                    hs = CUTEstModel(name)
                    test_name = name * " problem resolution"

                    @testset "$test_name optimality via ipopt" begin
                        
                        resol = NLPModelsIpopt.ipopt(hs, max_iter = 5000, print_level= ((name == "HS49") | (name == "HS55")) ? 5 : 0, tol = ω, constr_viol_tol = η, compl_inf_tol = ϵ)
                        
                        if (name == "HS13") | (name == "HS49") | (name == "HS55")
                            @test_broken KKT_check(hs, resol.solution, - resol.solver_specific[:multipliers_con] , resol.solver_specific[:multipliers_U] , resol.solver_specific[:multipliers_L] , ω, η, ϵ, 7)
                        else
                            @test KKT_check(hs, resol.solution, - resol.solver_specific[:multipliers_con] , resol.solver_specific[:multipliers_U] , resol.solver_specific[:multipliers_L] , ω, η, ϵ, 0)
                        end

                    end
                    finalize(hs)
                end
            end


            @testset "NCLSolve NLP (only linear residuals)" begin

                @testset "KKT_check function" begin
                    @testset "KKT_check(nlp) via ipopt" begin
                        # Resolution of NLP with NLPModelsIpopt
                            resol_nlp_ipopt = NLPModelsIpopt.ipopt(nlp, print_level = 0, tol = ω, constr_viol_tol = η, compl_inf_tol = ϵ, ignore_time = true)
                            x_nlp_ipopt = resol_nlp_ipopt.solution
                            
                            # Get multipliers
                            λ_nlp_ipopt = - resol_nlp_ipopt.solver_specific[:multipliers_con]
                            z_U_nlp_ipopt = resol_nlp_ipopt.solver_specific[:multipliers_U]
                            z_L_nlp_ipopt = resol_nlp_ipopt.solver_specific[:multipliers_L]

                        @test_broken KKT_check(nlp, [0.5, 1.0], [1., 0., 0., -2.0], [0, 1.], [0., 0.0], ω, η, ϵ, print_level) # solved by hand
                        @test KKT_check(nlp, x_nlp_ipopt, λ_nlp_ipopt, z_U_nlp_ipopt, z_L_nlp_ipopt, ω, η, ϵ, print_level)
                    end

                    @testset "KKT_check(ncl_nlin_res) via ipopt" begin
                        # Resolution of ncl_nlin_res with NLPModelsIpopt
                            resol_ncl_ipopt = NLPModelsIpopt.ipopt(ncl_nlin_res, print_level = 0, tol = ω, constr_viol_tol = η, compl_inf_tol = ϵ, ignore_time = true)
                            x_ncl_ipopt = resol_ncl_ipopt.solution
                            
                            # Get multipliers
                            λ_ncl_ipopt = - resol_ncl_ipopt.solver_specific[:multipliers_con]
                            z_U_ncl_ipopt = resol_ncl_ipopt.solver_specific[:multipliers_U]
                            z_L_ncl_ipopt = resol_ncl_ipopt.solver_specific[:multipliers_L]

                        @test KKT_check(ncl_nlin_res, x_ncl_ipopt, λ_ncl_ipopt, z_U_ncl_ipopt, z_L_ncl_ipopt, ω, η, ϵ, print_level)
                    end

                    
                end

                @testset "KKT_check(nlp) via NCLSolve" begin
                    # Resolution of ncl_nlin_res with NCL method
                        resol_ncl_ncl = NCLSolve(ncl_nlin_res, max_iter_NCL = 30, use_ipopt = true, tol = ω, constr_viol_tol = η, compl_inf_tol = ϵ, print_level = print_level)
                        x_ncl = resol_ncl_ncl.solution

                        λ_ncl = resol_ncl_ncl.solver_specific[:multipliers_con]
                        z_U_ncl = resol_ncl_ncl.solver_specific[:multipliers_U]
                        z_L_ncl = resol_ncl_ncl.solver_specific[:multipliers_L]    
                    
                    @test KKT_check(nlp, x_ncl[1:ncl_nlin_res.nvar_x], λ_ncl, z_U_ncl[1:ncl_nlin_res.nvar_x], z_L_ncl[1:ncl_nlin_res.nvar_x], ω, η, ϵ, print_level) 
                end
            end


            @testset "NCLSolve NLP (all residuals)" begin
                @testset "KKT_check(nlc_cons_res) via ipopt" begin
                    # Resolution of ncl_nlin_res with NLPModelsIpopt
                        resol_ncl_ipopt = NLPModelsIpopt.ipopt(nlc_cons_res, print_level = 0, tol = ω, constr_viol_tol = η, compl_inf_tol = ϵ, ignore_time = true)
                        x_ncl_ipopt = resol_ncl_ipopt.solution
                        
                        # Get multipliers
                        λ_ncl_ipopt = - resol_ncl_ipopt.solver_specific[:multipliers_con]
                        z_U_ncl_ipopt = resol_ncl_ipopt.solver_specific[:multipliers_U]
                        z_L_ncl_ipopt = resol_ncl_ipopt.solver_specific[:multipliers_L]

                    @test KKT_check(nlc_cons_res, x_ncl_ipopt, λ_ncl_ipopt, z_U_ncl_ipopt, z_L_ncl_ipopt, ω, η, ϵ, print_level)
                end

                @testset "KKT_check(nlp) via NCLSolve" begin
                    # Resolution of nlc_cons_res with NCL method
                        resol_ncl_ncl = NCLSolve(nlc_cons_res, max_iter_NCL = 30, use_ipopt = true, tol = ω, constr_viol_tol = η, compl_inf_tol = ϵ, print_level = print_level)
                        x_ncl = resol_ncl_ncl.solution
                
                        λ_ncl = resol_ncl_ncl.solver_specific[:multipliers_con]
                        z_U_ncl = resol_ncl_ncl.solver_specific[:multipliers_U]
                        z_L_ncl = resol_ncl_ncl.solver_specific[:multipliers_L]

                    @test KKT_check(nlp, x_ncl[1:nlc_cons_res.nvar_x], λ_ncl, z_U_ncl[1:nlc_cons_res.nvar_x], z_L_ncl[1:nlc_cons_res.nvar_x], ω, η, ϵ, print_level) 
                end
            end

            @testset "NCLSolve HS (only linear residuals)" begin
                for name in probs_NCL # several tests
                    nlp = CUTEstModel(name)
                    test_name = name * " problem resolution"
                    @testset "$test_name" begin
                        @test NCLSolve(nlp, max_iter_NCL = 20, print_level=0).solver_specific[:internal_msg] == Symbol("Solve_Succeeded")
                    end
                    finalize(nlp)
                end
            end

            @testset "NCLSolve HS (only linear residuals)" begin
                for name in probs_NCL # several tests
                    nlp = CUTEstModel(name)
                    test_name = name * " problem resolution"
                    @testset "$test_name" begin
                        @test NCLSolve(nlp, max_iter_NCL = 20, print_level=0, linear_residuals=false).solver_specific[:internal_msg] == Symbol("Solve_Succeeded")
                    end
                    finalize(nlp)
                end
            end

            @testset "NCLSolve HS (all residuals)" begin
                for name in probs_NCL # several tests
                    nlp = CUTEstModel(name)
                    test_name = name * " problem resolution"
                    @testset "$test_name" begin
                        @test NCLSolve(nlp, max_iter_NCL = 20, print_level=0, linear_residuals = true).solver_specific[:internal_msg] == Symbol("Solve_Succeeded")
                    end
                    finalize(nlp)
                end
            end

        else
            @testset "Empty test" begin
                @test true
            end
        end
    end
    #############################
    #############################




    """
    ################################
    Little fonction to do both tests
    ################################
    """
    function test_main(test_NCLModel_command::Bool, test_NCLSolve_command::Bool) ::Test.DefaultTestSet
        test_NLCModel(test_NCLModel_command)
        test_NCLSolve(test_NCLSolve_command)
    end
    ################################
    ################################
    test_main(true,true)


#####################