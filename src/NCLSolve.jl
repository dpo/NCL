# comment
#** Important
# ! Warning / Problem
# ? Question
# TODO

using LinearAlgebra
using Printf

using NLPModels
using SolverTools
using NLPModelsIpopt

include("NCLModel.jl")
include("KKT_check.jl")

#using NLPModelsKnitro

#! TODO Fix closing file problem...

######### TODO #########
######### TODO #########
######### TODO #########

    # TODO (feature)   : Créer un vrai statut
    # TODO KKT_check output in file to fix

    # TODO (recherche) : choix des mu_init à améliorer...
    # TODO (recherche) : Points intérieurs à chaud...
    # TODO (recherche) : tester la proximité des multiplicateurs y_k de renvoyés par le solveur et le ncl.y du problème (si r petit, probablement proches.)
    # TODO (recherche) : Mieux choisir le pas pour avoir une meilleure convergence
    # TODO (recherche) : ajuster eta_end

    # TODO (Plus tard) : Pierric pour choix de alpha, beta, tau...
########## TODO ########
########## TODO ########
########## TODO ########




# NCLSolve called with a string outputs to file
function NCLSolve(nlp::AbstractNLPModel, file::String; kwargs...)
    out = open(file, "w") do io
        NCLSolve(nlp; io=io, kwargs...)
    end
    
    return out
end

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
                SolverTools.status                              # of the last solve
                ncl                                             # the problem in argument,
                solution = sol,                                 # solution found
                iter = k,                                       # number of iteration of the ncl method (not iteration to solve subproblems)
                objective=obj(ncl, sol),                        # objective value
                elapsed_time=0,                                 # time of computation of the whole solve
                solver_specific=Dict(:multipliers_con => y_k,   # lagrangian multipliers for : constraints
                                    :multipliers_L => z_k_L,    #                              upper bounds
                                    :multipliers_U => z_k_U     #                              lower bounds
                                    )
                )
#################################
"""
function NCLSolve(nlp::AbstractNLPModel ;                    # Problem to be solved by this method
                  #* Optimization parameters
                  tol::Float64 = 1e-6,                       # Tolerance for the gradient lagrangian norm
                  constr_viol_tol::Float64 = 1e-6,           # Tolerance for the infeasibility accepted for ncl
                  compl_inf_tol::Float64 = 0.0001,           # Tolerance for the complementarity accepted for ncl
                  acc_factor::Float64 = 100.,

                  #* Options for NCL
                  max_penal::Float64 = 1e15,                 # Maximal penalization authorized (ρ_max)
                  min_infeas::Float64 = 1e-10,               # Minimal infeasibility authorized (η_min)
                  max_iter_NCL::Int64 = 20,                    # Maximum number of iterations for the NCL method
                  linear_residuals::Bool = false,            # To choose if you want to put residuals upon linear constraints or not
                  KKT_checking::Bool = false,                # To choose if you want to check KKT conditions in the loop, or just stop when residuals are small enough.

                  #* Options for solver
                  max_iter_solver::Int64 = 1000,               # Maximum number of iterations for the subproblem solver
                  print_level_solver::Int64 = 0,               # Options for printing iterations of the subproblem solver
                  warm_start_init_point::String = "yes",     # "yes" to choose warm start in the subproblem solving. "no" for normal solving.
 
                  #* Options of NCL print
                  io::IO = stdout,                             # where to print iterations
                  print_level_NCL::Int64 = 0,                  # Options for printing iterations of the NCL method : 0, nothing;
                                                                                                                    # 1, calls to functions and conclusion;
                                                                                                                    # 2, calls, little informations on iterations;
                                                                                                                    # 3, calls, more information about iterations (and erors in KKT_check);
                                                                                                                    # 4, calls, KKT_check, iterations, little information from the solver;                                                                                                                         # and so on until 7 (no further details)
                  
                 ) ::GenericExecutionStats                   # See NLPModelsIpopt / NLPModelsKnitro and SolverTools for further details on this structure
 
    #** I.0 Solution with NCL 
    if nlp isa NCLModel #no need to pass through NCLModel constructor
        ncl = nlp
    else
        ncl = NCLModel(nlp;
                       res_val_init = 0.,
                       res_lin_cons = linear_residuals)
    end
    if (nlp.meta.ncon == 0) | ((nlp.meta.nnln == 0) & !linear_residuals) # No need to create an NCLModel, because it is an unconstrained problem or it doesn't have non linear constraints
        no_res = true
        nr = 0
        nx = ncl.meta.nvar
    else
        no_res = false
        nr = ncl.nr
        nx = ncl.nx
    end

    #** I. Names and variables
    #** I.1 Constants & scale parameters
    warm_start = (warm_start_init_point == "yes")
    mu_init = warm_start ? 1.0e-3 : 0.1

    τ = 10.0 # scale (used to update the ρ_k step)
    τ_inv = 1.0/τ
    α = 0.1 # Constant (α needs to be < 1)
    β = 0.2 # Constant

    #** I.2 Parameters
    if !no_res
        ncl.ρ = 100.0 # step
    end
    ρ_max = max_penal # biggest penalization authorized

    ω_end = tol #global tolerance, in argument
    ω_k = 1e-2 # sub problem initial tolerance

    #! change eta_end
    η_end = constr_viol_tol #global infeasibility in argument
    η_k = 1e-2 # sub problem infeasibility
    η_min = min_infeas # smallest infeasibility authorized

    ϵ_end = compl_inf_tol #global tolerance for complementarity conditions

    # TODO: \epsilon_k a créer, décroissant, comme \eta_k.

    if no_res 
        ω_k = ω_end
        η_k = η_end
    end

    acc_count = 0


    #** I.3 Solver parameters
    acceptable_tol::Float64 = acc_factor * tol
    acceptable_constr_viol_tol::Float64 = acc_factor * constr_viol_tol
    acceptable_compl_inf_tol::Float64 = acc_factor * compl_inf_tol

    output_file_name_solver = "ncl_$(ncl.meta.name).log"

    #** I.4 Initial variables
    x_k = zeros(Float64, nx)
    r_k = zeros(Float64, nr)
    norm_r_k_inf = norm(r_k, Inf) #Pre-computation

    y_k = zeros(Float64, ncl.meta.ncon)
    z_k_U = zeros(Float64, length(ncl.meta.uvar))
    z_k_L = zeros(Float64, length(ncl.meta.lvar))

    #** I.5 Initial print
    if print_level_NCL ≥ 1
        @printf(io, "=== %s ===\n", ncl.nlp.meta.name)

        if print_level_NCL ≥ 2
            @printf(io, "Optimization parameters")
                @printf(io, "\n    Global tolerance                 ω_end = %7.2e for gradient lagrangian norm", tol)
                @printf(io, "\n    Global infeasibility             η_end = %7.2e for residuals norm and constraint violation", constr_viol_tol)
                @printf(io, "\n    Global complementarity tolerance ϵ_end = %7.2e for multipliers and constraints", compl_inf_tol)
                @printf(io, "\n    Maximum penalty parameter        ρ_max = %7.2e ", max_penal)
                @printf(io, "\n    Minimum infeasibility accepted   η_min = %7.2e \n", min_infeas)
            
            @printf(io, "============= NCL Begin =============\n")
            @printf(io, "%4s  %7s  %7s  %7s  %7s  %7s  %9s  %7s  %7s  %7s",
                        "Iter", "‖rₖ‖∞", "ηₖ", "ωₖ", "ρ", "μ init", "NCL obj", "‖y‖", "‖yₖ‖", "‖xₖ‖")
        end
    end

    print_level_solver > 0 && @printf(io, "\n")

    #** II. Optimization loop and return
    k = 0
    converged = false

    while (k ≤ max_iter_NCL) & !converged
        #** II.0 Iteration counter and mu_init
        k += 1
        
        if (k >= 3) & no_res #no residuals but still in the loop
            error("in NCLSolve($(ncl.meta.name)): problem $(ncl.meta.name) is unconstrained but ipopt did not solve it at acceptable level at the first time.\nYour problem is probably degenerated")
        end

        if (k == 2) & warm_start
            mu_init = 1e-4
        elseif (k == 4) & warm_start
            mu_init = 1e-5
        elseif (k == 6) & warm_start
            mu_init = 1e-6
        elseif (k == 8) & warm_start
            mu_init = 1e-7
        elseif (k == 10) & warm_start
            mu_init = 1e-8
        end

        #** II.1 Get subproblem's solution
        #** II.1.1 Solver
        solve_k = ipopt(ncl;
                        tol = ω_k,
                        constr_viol_tol = η_k,
                        compl_inf_tol = ϵ_end,
                        acceptable_tol = acceptable_tol,
                        acceptable_constr_viol_tol = acceptable_constr_viol_tol,
                        acceptable_compl_inf_tol = acceptable_compl_inf_tol,
                        print_level = print_level_solver,
                        output_file = print_level_solver > 0 ? output_file_name_solver : "",
                        ignore_time = true,
                        warm_start_init_point = warm_start_init_point,
                        mu_init = mu_init,
                        dual_inf_tol=1e-6,
                        max_iter = 1000,
                        sb = "yes")

        # Get variables
        X_k = solve_k.solution #pre-access
        x_k = X_k[1:nx]
        r_k = X_k[nx+1 : nx+nr]
        norm_r_k_inf = norm(r_k, Inf) # update

        # Get multipliers
        #! Warning, ipopt doesn't use our convention in KKT_check for constraint multipliers, so we took the opposite. For bound multiplier it seems to work though.
        y_k = - solve_k.solver_specific[:multipliers_con]
        z_k_U = solve_k.solver_specific[:multipliers_U]
        z_k_L = solve_k.solver_specific[:multipliers_L]

        #** II.1.2 Output print
        if print_level_NCL ≥ 2
            @printf(io, "%4d  %7.1e  %7.1e  %7.1e  %7.1e  %7.1e  %9.2e  %7.1e  %7.1e  %7.1e",
                        k, norm_r_k_inf, η_k, ω_k, ncl.ρ, mu_init, obj(ncl, vcat(x_k, r_k)), norm(ncl.y, Inf), norm(y_k, Inf), norm(x_k))
        end

        print_level_solver > 0 && @printf(io, "\n")

        #** II.2 Treatment & update
        if (norm_r_k_inf ≤ max(η_k, η_end)) | (k == max_iter_NCL) # The residual has decreased enough
            #** II.2.1 Update
            if !no_res
                ncl.y = ncl.y + ncl.ρ * r_k # Updating multiplier
            end
            η_k = max(η_k*τ_inv, η_min) # η_k / (1 + ncl.ρ ^ β) # (heuristic)
            ω_k = ω_k*τ_inv # TODO optimiser, pas de division inutile

            if η_k == η_min
                @warn "in NCLSolve($(ncl.meta.name)): minimum constraint violation η_min = " * string(η_min) * " reached at iteration k = " * string(k)
            end

            #** II.2.2 Solution found ?
            if (norm_r_k_inf ≤ η_end) | (k == max_iter_NCL) # check if r_k is small enough, or if we've reached the end
                #* Residual & KKT tests
                if norm_r_k_inf > η_end
                    converged = false
                else
                    if print_level_NCL ≥ 2
                        @printf(io, "\n‖rₖ‖∞ = %7.1e ≤ η_end = %7.1e", norm_r_k_inf, η_end)
                    end

                    if KKT_checking
                        if no_res
                            D_solved = KKT_check(ncl, x_k, y_k, z_k_U[1:nx], z_k_L[1:nx] ; io=io, tol=ω_end, constr_viol_tol=η_end, compl_inf_tol=ϵ_end, print_level=print_level_NCL)
                        else
                            D_solved = KKT_check(ncl.nlp, x_k, y_k, z_k_U[1:nx], z_k_L[1:nx] ; io=io, tol=ω_end, constr_viol_tol=η_end, compl_inf_tol=ϵ_end, print_level=print_level_NCL)
                        end
                        converged = D_solved["optimal"]

                        if D_solved["acceptable"]
                            acc_count += 1 # if we are still on an acceptable level
                        else
                            acc_count = 0 # if not, then go back to 0
                        end
                    else
                        converged = true #Chose not to pass into KKT_check
                    end
                end

                status = solve_k.status
                if no_res
                    dual_feas = (ncl.meta.ncon != 0) ? norm(grad(ncl, x_k) - jtprod(ncl, x_k, y_k) - (z_k_L - z_k_U)[1:nx], Inf) : norm(grad(ncl, x_k) - (z_k_L - z_k_U)[1:nx], Inf)
                    primal_feas = norm(setdiff(vcat(cons(nlp, x_k) - nlp.meta.lcon, nlp.meta.ucon - cons(nlp, x_k)), [Inf, -Inf]), Inf)
                else
                    dual_feas = (ncl.meta.ncon != 0) ? norm(grad(ncl.nlp, x_k) - jtprod(ncl.nlp, x_k, y_k) - (z_k_L - z_k_U)[1:nx], Inf) : norm(grad(ncl.nlp, x_k) - (z_k_L - z_k_U)[1:nx], Inf)
                    primal_feas = norm(setdiff(vcat(cons(ncl.nlp, x_k) - ncl.nlp.meta.lcon, ncl.nlp.meta.ucon - cons(ncl.nlp, x_k)), [Inf, -Inf]), Inf)
                end

                #* Print results
                if print_level_NCL ≥ 1
                    @printf(io, "\n")
                    if converged
                        write(io, "EXIT: optimal solution found\n")
                    elseif acc_count ≥ 3
                        write(io, "EXIT: solved to acceptable level\n")
                    elseif k == max_iter_NCL
                        write(io, "EXIT: reached max_iter_NCL\n")
                    end

                    @printf(io, "============= NCL End =============\n")
                end

                #** II.2.3 Return if end of the algorithm
                if converged | (k == max_iter_NCL)
                    return GenericExecutionStats(status, ncl,
                                                solution = x_k,
                                                iter = k,
                                                primal_feas = primal_feas,
                                                dual_feas = dual_feas,
                                                objective = no_res ? obj(ncl, x_k) : obj(ncl.nlp, x_k),
                                                elapsed_time = 0,
                                                #! doesn't work... counters = nlp.counters,
                                                solver_specific = Dict(:multipliers_con => y_k,
                                                                       :multipliers_L => z_k_L[1:nx],
                                                                       :multipliers_U => z_k_U[1:nx],
                                                                       :internal_msg => converged ? Symbol("Solve_Succeeded") : Symbol("Solve_Failed"),
                                                                       :residuals => ncl isa NCLModel ? r_k : Float64[]
                                                                      )
                                                )
                #else
                #    ncl.ρ = ncl.ρ * τ #! A voir !
                end
            end

        else
   #** II.3 Increase penalization
            ncl.ρ = τ * ncl.ρ # increase the step
            #η_k = η_end / (1 + ncl.ρ ^ α) # Change infeasibility (heuristic)
            if ncl.ρ == ρ_max
                @warn "in NCLSolve($(ncl.meta.name)): maximum penalty ρ = " * string(ρ_max) * " reached at iteration k = " * string(k)
            end
        end
        # ? Chez Nocedal & Wright, p.521, on a : ω_k = 1/ncl.ρ, ncl.ρ = 100ρ_k, η_k = 1/ncl.ρ^0.1
    end
end
