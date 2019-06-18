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


#! TODO Fix closing file problem...



######### TODO #########
######### TODO #########
######### TODO #########

    # TODO (feature)   : Créer un vrai statut
    # TODO (feature)   : faire le print dans un fichier externe
    # TODO (feature)   : print de tableau LaTeX aussi

    # TODO (clarify)   : Affichage output file...

    # TODO (recherche) : choix des mu_init à améliorer...
    # TODO (recherche) : Points intérieurs à chaud...
    # TODO (recherche) : tester la proximité des multiplicateurs λ_k de renvoyés par le solveur et le ncl.y du problème (si r petit, probablement proches.)
    # TODO (recherche) : Mieux choisir le pas pour avoir une meilleure convergence
    # TODO (recherche) : update ω_k 
    # TODO (recherche) : ajuster eta_end

    # TODO (Plus tard) : Pierric pour choix de alpha, beta, tau...
    # TODO (Plus tard) : print de tableau LaTeX aussi

########## TODO ########
########## TODO ########
########## TODO ########











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
function mult_format_check(z_U::Vector{<:Float64}, z_L::Vector{<:Float64}, ϵ::Float64) ::Tuple{Vector{<:Float64}, Vector{<:Float64}}
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
function KKT_check(nlp::AbstractNLPModel,                          # Problem considered
                   #* Position en multipliers
                     x::Vector{<:Float64},                           # Potential solution
                     λ::Vector{<:Float64},                           # Lagrangian multiplier for constraint
                     z_U::Vector{<:Float64},                         # Lagrangian multiplier for upper bound constraint
                     z_L::Vector{<:Float64};                         # Lagrangian multiplier for lower bound constraint
  
                   #* Tolerances
                     tol::Float64 = 0.001,                                     # Tolerance for lagrangien gradient norm
                     constr_viol_tol::Float64 = 0.0001,                                     # Tolerance or constraint violation
                     compl_inf_tol::Float64 = 0.0001,                                     # Tolerance for complementarity
  
                   #* Print options
                     print_level::Int64 = 0,                         # Verbosity of the function : 0 : nothing
                                                                                              # 1 : Function call and result
                                                                                              # 2 : Further information in case of failure
                                                                                              # 3... : Same, increasing information
                                                                                              # 6 & 7 : Shows full vectors, not advised if your problem has a big size
                     output_file_print::Bool = true,                 # Choose to print in an output file or stdout
                     output_file_name::String = "KKT_check.log",
                     output_file::IOStream = open("KKT_check.log", write = true)  # Path until file for printing details
                   ) ::Bool                                       # true returned if the problem is solved at x, with tolerances specified. false instead.

    #** 0. Initial settings
        #** 0.1 Notations
            ω = tol
            η = constr_viol_tol
            ϵ = compl_inf_tol


        #** 0.2 Print
            if print_level >= 1 # Translation : We will print smthg, 
                if output_file_print # in a file
                    if output_file_name == "KKT_check.log" #If the name is the default one, then we take the openned file (and won't close it at the end)
                        file = output_file 
                    else # if it is not the default one
                        file = open(output_file_name, write = true) # then we open this file and then close it at the end.
                    end
                else
                    file = stdout # if not in a file, in the terminal
                end
            end
            

            if print_level >= 1
                @printf(file, "\nKKT_check called on %s \n", nlp.meta.name)
            end

        #** 0.3 Warnings et format
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

                            if output_file_print & (output_file_name != "KKT_check.log")
                                close(file)
                            end
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
                            
                            if output_file_print & (output_file_name != "KKT_check.log")
                                close(file)
                            end
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
                            
                            if output_file_print & (output_file_name != "KKT_check.log")
                                close(file)
                            end
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
                        
                        if output_file_print & (output_file_name != "KKT_check.log")
                            close(file)
                        end
                    end

                    return false 
                end
            end

        #** II.2 Complementarity
            for i in 1:nlp.meta.ncon # upper constraints
                if !( (-ϵ <= (λ[i] * (c_x[i] - nlp.meta.ucon[i])) <= ϵ)  |  (-ϵ <= (λ[i] * (c_x[i] - nlp.meta.lcon[i])) <= ϵ) )  # Complementarity condition (for range constraint, we have necessarily : [λ[i] * (c_x[i] - nlp.meta.lcon[i])] * [λ[i] * (c_x[i] - nlp.meta.ucon[i])] = 0
                    if print_level >= 1
                        if print_level >= 2
                            @printf(file, "    one of the two complementarities %7.2e or %7.2e is out of tolerance ϵ = %7.2e. See cons %d \n", λ[i] * (c_x[i] - nlp.meta.ucon[i]), (λ[i] * (c_x[i] - nlp.meta.lcon[i])), ϵ, i)

                            if print_level >= 3
                                @printf(file, "      λ[%d]             = %7.2e \n", i, λ[i])
                                @printf(file, "      c_x[%d]           = %7.2e \n", i, c_x[i])
                                @printf(file, "      nlp.meta.ucon[%d] = %7.2e \n", i, nlp.meta.ucon[i])
                                @printf(file, "      nlp.meta.lcon[%d] = %7.2e \n", i, nlp.meta.lcon[i])
                            end
                        end

                        write(file, "\n  ------- Not fitting with KKT conditions ----------\n")
                        
                        if output_file_print & (output_file_name != "KKT_check.log")
                            close(file)
                        end
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
                                    @printf(file, "      ||∇f_x||                = %7.2e \n", norm(∇f_x, Inf))
                                    @printf(file, "      ||∇f_x - t(Jac_x) * λ|| = %7.2e \n", norm(∇f_x - jtprod(nlp, x, λ), Inf))
                                    @printf(file, "      ||z||                   = %7.2e \n", norm(z, Inf))
                                    @printf(file, "      ||∇lag_x||              = %7.2e \n", norm(∇lag_x, Inf))
                                else
                                    @printf(file, "      ||∇f_x||   = %7.2e \n", norm(∇f_x, Inf))
                                    @printf(file, "      ||- z||    = %7.2e \n", norm(z, Inf))
                                    @printf(file, "      ||∇lag_x|| = %7.2e \n", norm(∇lag_x, Inf))
                                end
                            end
                        end  
                    end
                
                    write(file, "\n  ------- Not fitting with KKT conditions ----------\n")
                    
                    if output_file_print & (output_file_name != "KKT_check.log")
                        close(file)
                    end
                end
                
                return false
            end
        
    #** IV Return if tests were passed
            if print_level >= 1
            @printf(file, "    %s problem solved !\n", nlp.meta.name)
        end

        if output_file_print & (output_file_name != "KKT_check.log")
            close(file)
        end
        return true # all the tests were passed, x, λ respects feasability, complementarity respected, and ∇lag_x(x, λ) almost = 0
end

#######################


















"""
#################################
Main function for the NCL method.
    Takes an AbstractNLPModel as initial problem, 
    Converts it to a NCLModel,
    Runs the following NCL method on it
        See https://www.researchgate.net/publication/325480151_Stabilized_Optimization_Via_an_NCL_Algorithm for further explications on the method
        Arguments: 
            nlp: optimization problem described by the modelization NLPModels.jl (voir https://github.com/JuliaSmoothOptimizers/NLPModels.jl)
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
    Returns (x (solution), y (lagrangian multipliers for constraints), z (lagrangian multpliers for bound constraints))
#################################
"""
function NCLSolve(nlp::AbstractNLPModel;                                               # Problem to be solved by this method (see NCLModel.jl for further details)
                 #* Many key-word arguments
                   #* Optimization parameters
                     tol::Float64 = 1e-6,                                                # Tolerance for the gradient lagrangian norm
                     constr_viol_tol::Float64 = 1e-6,                                     # Tolerance for the infeasability accepted for ncl
                     compl_inf_tol::Float64 = 0.0001,                                     # Tolerance for the complementarity accepted for ncl
                     linear_residuals::Bool = true,                                       # Boolean to choose if you want residuals onlinear constraints (true), or not (false)
     
                   #* Options for NCL
                     max_penal::Float64 = 1e15,                                           # Maximal penalization authorized (ρ_max)
                     min_infeas::Float64 = 1e-8,                                          # Minimal infeasability authorized (η_min)
                     max_iter_NCL::Int64 = 20,                                            # Maximum number of iterations for the NCL method
                     print_level_NCL::Int64 = 0,                                          # Options for printing iterations of the NCL method : 0, nothing; 
                                                                                                                                           # 1, calls to functions and conclusion; 
                                                                                                                                           # 2, calls, little informations on iterations; 
                                                                                                                                           # 3, calls, more information about iterations (and erors in KKT_check); 
                                                                                                                                           # 4, calls, KKT_check, iterations, little information from the solver; 
                                                                                                                                           # and so on until 7 (no further details)
                     output_file_print_NCL::Bool = false,                                 # Choose if you want NCL iterations in a separate file
                     output_file_name_NCL::String = "NCL.log",                            # The name of the separate file
                     output_file_NCL::IOStream = open("NCL.log", write=true),             # The file itself, openned, if you like. Only if you want to include it in another file. Won't be closed at the end
                     KKT_checking::Bool = true,                                           # To choose if you want to check KKT conditions in the loop, or just stop when residuals are small enough.

                   #* Options for solver 
                     use_ipopt::Bool = true,                                              # Boolean to chose the solver you want to use (true => IPOPT, false => KNITRO)
                     max_iter_solver::Int64 = 1000,                                       # Maximum number of iterations for the subproblem solver
                     print_level_solver::Int64 = 0,                                       # Options for printing iterations of the subproblem solver
                     output_file_print_solver::Bool = false,                              # Choose if you want subproblems iterations in a separate file
                     output_file_name_solver::String = "subproblem_solver.log",           # The name of the separate file
                     warm_start_init_point::String = "yes",                               # "yes" to choose warm start in the subproblem solving. "no" for normal solving.
                     
                 ) ::GenericExecutionStats                                              # See NLPModelsIpopt / NLPModelsKnitro and SolverTools for further details on this structure

    #** 0. Printing and file choices
        if print_level_NCL >= 1  # Translation : We will print smthg, 
            if output_file_print_NCL # in a file
                if output_file_name_NCL == "NCL.log"  #If the name is the default one, then we take the openned file (and won't close it at the end)
                    file = output_file_NCL
                else # if it is not the default one
                    file = open(output_file_name_NCL, write = true) # then we open this file and then close it at the end.
                end
            else
                file = stdout # if not in a file, in the terminal
            end

            @printf(file, "=== %s resolution ===\n", nlp.meta.name)

            if print_level_NCL >= 2
                @printf(file, "Optimization parameters")
                    @printf(file, "\n    Global tolerance                 ω_end = %7.2e for gradient lagrangian norm", tol)
                    @printf(file, "\n    Global infeasability             η_end = %7.2e for residuals norm and constraint violation", constr_viol_tol)
                    @printf(file, "\n    Global complementarity tolerance ϵ_end = %7.2e for multipliers and constraints", compl_inf_tol)
                    @printf(file, "\n    Maximal penalization accepted    ρ_max = %7.2e ", max_penal)
                    @printf(file, "\n    Minimal infeasability accepted   η_min = %7.2e \n", min_infeas)
            end

        end

    if (nlp.meta.ncon == 0) | ((nlp.meta.nnln == 0) & !linear_residuals)
        #** I. Resolution with solver
            if use_ipopt
                if print_level_NCL >= 1
                    @printf(file, "\n============= Resolution of %s with IPOPT (0 residual to add) =============\n", nlp.meta.name)
                end
                
                if (print_level_solver >= 1) & output_file_print_solver
                    resol = NLPModelsIpopt.ipopt(nlp ; 
                                                 print_level = print_level_solver, 
                                                 output_file = output_file_name_solver, 
                                                 max_iter = max_iter_solver, 
                                                 tol = tol, 
                                                 constr_viol_tol = constr_viol_tol, 
                                                 compl_inf_tol = compl_inf_tol)
                else
                    resol = NLPModelsIpopt.ipopt(nlp ; 
                                                 print_level = print_level_solver, 
                                                 max_iter = max_iter_solver, 
                                                 tol = tol, 
                                                 constr_viol_tol = constr_viol_tol, 
                                                 compl_inf_tol = compl_inf_tol)
                end
                
                if print_level_NCL >= 1
                    if print_level_NCL >= 2
                        if nlp.meta.ncon != 0
                            @printf(file, "    Initial lagrangian gradient : ||∇Lag(x, λ)|| = %7.2e (tolerance = %7.1e)\n", norm(grad(nlp, nlp.meta.x0) - jtprod(nlp, nlp.meta.x0, nlp.meta.y0), Inf), tol)
                        else
                            @printf(file, "    Initial lagrangian gradient : ||∇Lag(x, λ)|| = %7.2e (tolerance = %7.1e)\n", norm(grad(nlp, nlp.meta.x0), Inf), tol)
                        end
                    end

                    @printf(file, "    Solver conclusion: %s\n", string(resol.solver_specific[:internal_msg]))
                   
                    if print_level_NCL >= 2
                        @printf(file, "        Number of iteration: %d\n", resol.iter)
                        println(file, "        Final point: x = ", resol.solution)
                        @printf(file, "        Final lagrangian gradient: ||∇Lag(x, λ)|| = %7.2e (tolerance = %7.1e)\n", resol.dual_feas, tol)
                    end
                end

                return resol

            else
                if print_level_NCL >= 1
                    @printf(file, "Resolution of %s with KNITRO (because 0 residual to add)\n", nlp.meta.name)
                end
                
                return _knitro(nlp)
            end

    else

        #** II. Resolution with NCL 
            #** II.0 NCLModel ?
                if isa(nlp, NCLModel) # Need to add residuals ?
                    ncl = nlp 
                else
                    ncl = NCLModel(nlp ; 
                                print_level  = print_level_NCL,
                                res_val_init = 0., 
                                res_lin_cons = linear_residuals,
                                output_file_print = output_file_print_NCL,
                                output_file_name = output_file_name_NCL,
                                output_file = output_file_NCL
                                )
                end

            
            #** II.A. Names and variables
                #** II.A.0 Constants & scale parameters
                    Type::DataType = typeof(ncl.meta.x0[1])
                    warm::Bool = (warm_start_init_point == "yes") # Little precomputation
                    mu_init::Float64 = 1.

                    if warm
                        mu_init = 1e-3
                    else
                        mu_init = 0.1
                    end
                    
                    τ::Float32 = 10.0 # scale (used to update the ρ_k step)
                    α::Float32 = 0.1 # Constant (α needs to be < 1)
                    β::Float32 = 0.2 # Constant


                #** II.A.1 Parameters
                    ncl.ρ = 1f2 # step
                    ρ_max = max_penal # biggest penalization authorized

                    ω_end::Float64 = tol #global tolerance, in argument
                    ω_k::Float64 = ω_end # sub problem initial tolerance

                    #! change eta_end
                    η_end::Float64 = 1f-6 #constr_viol_tol #global infeasability in argument
                    η_k::Float64 = 1e-2 # sub problem infeasability
                    η_min = min_infeas # smallest infeasability authorized

                    ϵ_end = compl_inf_tol #global tolerance for complementarity conditions
                
        


                #** II.A.2 Initial variables
                    x_k = zeros(Type, ncl.nvar_x)
                    r_k = zeros(Type, ncl.nvar_r)
                    norm_r_k_inf = norm(r_k,Inf) #Pre-computation

                    λ_k = zeros(Type, ncl.meta.ncon)
                    z_k_U = zeros(Type, length(ncl.meta.uvar))
                    z_k_L = zeros(Type, length(ncl.meta.lvar))
                    

                #** II.A.3 Initial print
                    if print_level_NCL >= 1 
                        @printf(file, "\n============= Initial Model =============")
                            print(ncl, print_level = print_level_NCL, output_file_print = output_file_print_NCL, output_file = file)

                        if print_level_NCL >= 2
                            @printf(file, "\n\n======================================\n============= Iterations =============")
                                @printf(file, "\n%5s  %4s  %6s", "Iter", "||r_k||_{∞}", "η_k")
                                
                                if print_level_NCL >= 3
                                    @printf(file, " %7s  %11s  %13s", "ρ", "mu_init", "obj(ncl, X_k)")

                                    if print_level_NCL >= 6
                                        @printf(file, " %7s  %10s  %9s \n", "||y||", "||λ_k||", "||x_k||")
                                    else
                                        @printf(file, "\n")
                                    end

                                else
                                    @printf(file, "\n")
                                end
                        end
                        
                    end
            


            #** II.B. Optimization loop and return
                k = 0
                converged = false

                while (k <= max_iter_NCL) & !converged
                    #** II.B.0 Iteration counter and mu_init
                        k += 1
                        

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

                    #** II.B.1 Get subproblem's solution
                        #** II.B.1.1 Solver
                            if use_ipopt
                                if output_file_print_solver
                                    resolution_k = NLPModelsIpopt.ipopt(ncl ;
                                                                        tol = ω_k * 0.5, 
                                                                        constr_viol_tol = η_k * 0.5, 
                                                                        compl_inf_tol = ϵ_end * 0.5, 
                                                                        print_level = print_level_solver,
                                                                        output_file=output_file_name_solver,
                                                                        ignore_time = true, 
                                                                        warm_start_init_point = warm_start_init_point, 
                                                                        mu_init = mu_init, 
                                                                        dual_inf_tol=1e-6, 
                                                                        max_iter = 1000)
                                else
                                    resolution_k = NLPModelsIpopt.ipopt(ncl ;
                                                                        tol = ω_k, 
                                                                        constr_viol_tol = η_k, 
                                                                        #compl_inf_tol = ϵ_end, 
                                                                        print_level = print_level_solver,
                                                                        ignore_time = true, 
                                                                        warm_start_init_point = warm_start_init_point, 
                                                                        mu_init = mu_init, 
                                                                        dual_inf_tol=1e-6, 
                                                                        max_iter = 1000)
                                end
                                    # Get variables
                                    X_k = resolution_k.solution #pre-access
                                    x_k = X_k[1:ncl.nvar_x]
                                    r_k = X_k[ncl.nvar_x+1 : ncl.nvar_x+ncl.nvar_r]
                                    norm_r_k_inf = norm(r_k,Inf) # update

                                    # Get multipliers
                                    #! Warning, ipopt doesn't use our convention in KKT_check for constraint multipliers, so we took the opposite. For bound multiplier it seems to work though.
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
                        
                        #** II.B.1.2 Output print
                            if print_level_NCL >= 2 
                                if k % 10 == 0  # just reminding
                                    @printf(file, "\n%5s  %4s  %6s", "Iter", "||r_k||_{∞}", "η_k") 

                                    if print_level_NCL >= 3
                                        @printf(file, " %7s  %11s  %13s", "ρ", "mu_init", "obj(ncl, X_k)")

                                        if print_level_NCL >= 6
                                            @printf(file, " %7s  %10s  %9s \n", "||y||", "||λ_k||", "||x_k||")
                                        else
                                            @printf(file, "\n")
                                        end
                                    else
                                        @printf(file, "\n")
                                    end
                                end 

                                @printf(file, " %2d  %10.1e  %10.1e", k, norm_r_k_inf, η_k)
                        
                                if print_level_NCL >= 3
                                    @printf(file, " %9.1e  %8.1e  %9.1e", ncl.ρ, mu_init, obj(ncl, vcat(x_k, r_k)))

                                    if print_level_NCL >= 6
                                        @printf(file, " %12.1e  %9.1e  %9.1e \n", norm(ncl.y, Inf), norm(λ_k, Inf), norm(x_k))

                                        if print_level_NCL >= 7
                                            print(ncl ; print_level = print_level_NCL, current_X = vcat(x_k, r_k), output_file = output_file_NCL)
                                        end




                                    else
                                        @printf(file, "\n")
                                    end
                                    
                                else
                                    @printf(file, "\n")
                                end
                            end


                    #** II.B.2 Treatment & update
                        if (norm_r_k_inf <= max(η_k, η_end)) | (k == max_iter_NCL) # The residual has decreased enough
                            #** II.B.2.1 Update
                                ncl.y = ncl.y + ncl.ρ * r_k # Updating multiplier
                                η_k = max(η_k/τ, η_min) # η_k / (1 + ncl.ρ ^ β) # (heuristic)

                                if η_k == η_min
                                    @warn "Minimum constraint violation η_min = " * string(η_min) * " reached at iteration k = " * string(k)
                                end
                            
                            #** II.B.2.2 Solution found ?
                                if (norm_r_k_inf <= η_end) | (k == max_iter_NCL) # check if r_k is small enough, or if we've reached the end
                                    #* Residual & KKT tests
                                        if norm_r_k_inf > η_end
                                                converged = false
                                        else
                                            if print_level_NCL >= 2
                                                if KKT_checking
                                                    @printf(file, "--------\n   norm(r_k,Inf) = %7.2e  <= η_end = %7.2e. Calling KKT_check\n", norm_r_k_inf, η_end)
                                                else
                                                    @printf(file, "--------\n   norm(r_k,Inf) = %7.2e  <= η_end = %7.2e. Residual small enough, over\n", norm_r_k_inf, η_end)
                                                end
                                            end

                                            if KKT_checking
                                                if print_level_NCL >= 1
                                                    if output_file_print_NCL # Just to avoid type errors, not very important
                                                        converged = KKT_check(ncl.nlp, x_k, λ_k, z_k_U[1:ncl.nvar_x], z_k_L[1:ncl.nvar_x], tol=ω_end, constr_viol_tol=η_end, compl_inf_tol=ϵ_end, print_level=print_level_NCL, output_file_print = true, output_file = file)
                                                    end
                                                else
                                                    converged = KKT_check(ncl.nlp, x_k, λ_k, z_k_U[1:ncl.nvar_x], z_k_L[1:ncl.nvar_x], tol=ω_end, constr_viol_tol=η_end, compl_inf_tol=ϵ_end, print_level=print_level_NCL)
                                                end
                                            else
                                                converged = true #Chose not to pass into KKT_check
                                            end

                                        end
                                    
                                        status = resolution_k.status
                                        dual_feas::Float64 = (ncl.meta.ncon != 0) ? norm(grad(ncl.nlp, x_k) - jtprod(ncl.nlp, x_k, λ_k) - (z_k_U - z_k_L)[1:ncl.nvar_x], Inf) : norm(grad(ncl.nlp, x_k) - (z_k_U - z_k_L)[1:ncl.nvar_x], Inf)

                                    #* Print results
                                        if print_level_NCL >= 1
                                            if converged
                                                write(file, "\n==========================\n==========================\n",
                                                            "   EXIT: optimal solution found\n\n")
                                            elseif k == max_iter_NCL
                                                write(file, "\n==========================\n==========================\n",
                                                            "   EXIT: reached max_iter_NCL\n\n")
                                            end

                                            @printf(file, "============= Final Model =============")
                                            print(ncl, current_X = X_k, current_λ = λ_k, current_z = z_k_U - z_k_L, print_level = print_level_NCL, output_file_print = output_file_print_NCL, output_file = file)
                                        end
                                    
                            #** II.B.2.3 Return if end of the algorithm
                                    if converged | (k == max_iter_NCL)
                                        if (print_level_NCL >= 1) & output_file_print_NCL & (output_file_name_NCL != "NCL.log")
                                            close(file)
                                        end
                                        return GenericExecutionStats(status, nlp, 
                                                                    solution = x_k,
                                                                    iter = k,
                                                                    dual_feas = dual_feas,
                                                                    objective = obj(nlp, x_k), 
                                                                    elapsed_time = 0,
                                                                    solver_specific = Dict(:multipliers_con => λ_k,
                                                                                           :multipliers_L => z_k_L[1:ncl.nvar_x],
                                                                                           :multipliers_U => z_k_U[1:ncl.nvar_x],
                                                                                           :internal_msg => converged ? Symbol("Solve_Succeeded") : Symbol("Solve_Failed"),
                                                                                           :residuals => r_k
                                                                                          )
                                                                    )
                                    end
                                end

                        else 
                    #** II.B.3 Increase penalization
                            ncl.ρ = τ * ncl.ρ # increase the step 
                            #η_k = η_end / (1 + ncl.ρ ^ α) # Change infeasability (heuristic)
                            if ncl.ρ == ρ_max
                                @warn "Maximum penalization ρ = " * string(ρ_max) * " reached at iteration k = " * string(k)
                            end
                        end
                        # ? Chez Nocedal & Wright, p.521, on a : ω_k = 1/ncl.ρ, ncl.ρ = 100ρ_k, η_k = 1/ncl.ρ^0.1
                    
                end 
    end
end
#################################
