
export NCLSolve

# NCLSolve called with a string outputs to file
#function NCLSolve(nlp::AbstractNLPModel, file::String; kwargs...)
#    out = open(file, "w") do io
#        NCLSolve(nlp; io=io, kwargs...)
#    end
#
#    return out
#end

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
                  compl_inf_tol::Float64 = 1e-4,             # Tolerance for the complementarity accepted for ncl
                  acc_factor::Float64 = 100.,

                  #* Options for NCL
                  min_tol::Float64 = 1e-10,                  # Minimal tolerance authorized (ω_min)
                  min_constr_viol_tol::Float64 = 1e-10,      # Minimal infeasibility authorized (η_min)
                  max_penal::Float64 = 1e15,                 # Maximal penalization authorized (ρ_max)
                  min_compl_inf_tol::Float64 = 1e-8,

                  scale_penal::Float64 = 10.0,
                  scale_tol::Float64 = 0.1,
                  scale_constr_viol_tol::Float64 = 0.1,
                  scale_compl_inf_tol::Float64 = 0.1,
                  scale_mu_init::Float64 = 0.1,

                  init_penal::Float64 = 10.0,
                  init_tol::Float64 = 0.1,
                  init_constr_viol_tol::Float64 = 0.1,
                  init_compl_inf_tol::Float64 = 0.1,

                  max_iter_NCL::Int = 20,                  # Maximum number of iterations for the NCL method
                  linear_residuals::Bool = false,            # To choose if you want to put residuals upon linear constraints or not
                  KKT_checking::Bool = false,                # To choose if you want to check KKT conditions in the loop, or just stop when residuals are small enough.

                  #* Options for solver
                  max_iter_solver::Int = 1000,               # Maximum number of iterations for the subproblem solver
                  print_level_solver::Int = 0,               # Options for printing iterations of the subproblem solver
                  warm_start::Bool = true,     # "yes" to choose warm start in the subproblem solving. "no" for normal solving.

                  #* Options of NCL print
                  io::IO = stdout,                             # where to print iterations
                  print_level_NCL::Int = 0,                  # Options for printing iterations of the NCL method : 0, nothing;
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
    mu_init = warm_start ? 1.0e-3 : 0.1

    τ_ρ = scale_penal # scale (used to update the ρ_k penalization)
    τ_ϵ = scale_compl_inf_tol
    τ_η = scale_constr_viol_tol
    τ_ω = scale_tol
    τ_mu_init = scale_mu_init

    α = 0.1 # Constant (α needs to be < 1)
    β = 0.2 # Constant

    #** I.2 Parameters
    if !no_res
        ncl.ρ = 100.0 # penalization
    end
    ρ_max = max_penal # biggest penalization authorized

    ω_end = tol #global tolerance, in argument
    ω_k = init_tol # sub problem initial tolerance
    ω_min = min_tol

    #! change eta_end
    η_end = 0.5 * constr_viol_tol #global infeasibility in argument
    η_k = init_constr_viol_tol # sub problem infeasibility
    η_min = min_constr_viol_tol # smallest infeasibility authorized

    ϵ_end = compl_inf_tol #global tolerance for complementarity conditions
    ϵ_k = init_compl_inf_tol
    ϵ_min = min_compl_inf_tol


    if no_res
        ω_k = tol
        η_k = constr_viol_tol
        ϵ_k = compl_inf_tol
    end

    acc_count = 0
    iter_count = 0

    #** I.3 Solver acceptable parameters
    acceptable_tol::Float64 = acc_factor * ω_k
    acceptable_constr_viol_tol::Float64 = acc_factor * η_k
    acceptable_compl_inf_tol::Float64 = acc_factor * ϵ_k

    output_dir_tmp_solver = "/tmp/ncl/solver_logs"
    output_dir_solver = "./solver_logs"
    #if (!isdir(output_dir_solver)) & (print_level_solver > 0)
    #    mkdir(output_dir_solver)
    #elseif !isdir(output_dir_tmp_solver)
    #    mkdir(output_dir_tmp_solver)
    #    run(`chmod 700 /tmp/ncl/`) # only readable to me
    #end
    output_file_name_solver = print_level_solver > 0 ? "$(output_dir_solver)_ncl_$(ncl.meta.name).log" : "$(output_dir_tmp_solver)_tmp_ncl_$(ncl.meta.name).log"

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
                @printf(io, "\n    Maximum penalty parameter        ρ_max = %7.2e \n\n", max_penal)

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

        if (k >= 5) & no_res #no residuals but still in the loop
            error("\nin NCLSolve($(ncl.nlp.meta.name)): problem $(ncl.nlp.meta.name) is unconstrained but ipopt did not solve it at acceptable level at the first time.
                   \nYour problem is probably degenerated, or maybe you could raise an issue about it on github...")
        end

        warm_start && (mu_init *= τ_mu_init)

        #** II.1 Get subproblem's solution
        #** II.1.1 Solver
        solve_k = NLPModelsIpopt.ipopt(ncl;
                        tol = ω_k,
                        constr_viol_tol = η_k,
                        compl_inf_tol = ϵ_k,
                        acceptable_tol = acceptable_tol,
                        acceptable_constr_viol_tol = acceptable_constr_viol_tol,
                        acceptable_compl_inf_tol = acceptable_compl_inf_tol,
                        print_level = print_level_solver, #>= 1 ? print_level_solver : 2,
                        #output_file = output_file_name_solver,
                        ignore_time = true,
                        warm_start_init_point = warm_start_init_point,
                        mu_init = mu_init,
                        dual_inf_tol=1e-6,
                        max_iter = 1000)

        #(print_level_solver == 0) && rm(output_file_name_solver)


        # Get variables
        xr_k = solve_k.solution #pre-access
        x_k = xr_k[1:nx]
        r_k = xr_k[nx+1 : nx+nr]
        norm_r_k_inf = norm(r_k, Inf) # update

        # Get multipliers
        #! Warning, ipopt doesn't use our convention in KKT_check for constraint multipliers, so we took the opposite. For bound multiplier it seems to work though.
        y_k = - solve_k.solver_specific[:multipliers_con]
        z_k_U = solve_k.solver_specific[:multipliers_U]
        z_k_L = solve_k.solver_specific[:multipliers_L]

        iter_count += solve_k.iter + 1 #1 for NCL

        #** II.1.2 Output print
        if print_level_NCL ≥ 2
            @printf(io, "%4d  %7.1e  %7.1e  %7.1e  %7.1e  %7.1e  %9.2e  %7.1e  %7.1e  %7.1e",
                        k, norm_r_k_inf, η_k, ω_k, ncl.ρ, mu_init, obj(ncl, xr_k), norm(ncl.y, Inf), norm(y_k, Inf), norm(x_k))
        end

        print_level_solver > 0 && @printf(io, "\n")

        #** II.2 Treatment & update
        if (norm_r_k_inf ≤ max(η_k, η_end)) | (k == max_iter_NCL) # The residual has decreased enough
            #** II.2.1 Update
            if !no_res
                ncl.y = ncl.y + ncl.ρ * r_k # Updating multiplier
                η_k = max(η_k*τ_η, η_min) # η_k / (1 + ncl.ρ ^ β) # (heuristic)
                ϵ_k = max(ϵ_k*τ_ϵ, ϵ_min)
                ω_k = max(ω_k*τ_ω, ω_min)
            end

            if η_k == η_min
                @warn "\nin NCLSolve($(ncl.nlp.meta.name)): minimum constraint violation η_min = " * string(η_min) * " reached at iteration k = " * string(k)
            end
            if ω_k == ω_min
                @warn "\nin NCLSolve($(ncl.nlp.meta.name)): minimum tolerance ω_min = " * string(η_min) * " reached at iteration k = " * string(k)
            end
            if ϵ_k == ϵ_min
                @warn "\nin NCLSolve($(ncl.nlp.meta.name)): minimum complementarity infeasibility ϵ_min = " * string(η_min) * " reached at iteration k = " * string(k)
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
                        D_solved = KKT_check(ncl.nlp,
                                             x_k,
                                             y_k,
                                             z_k_U[1:nx],
                                             z_k_L[1:nx] ;
                                             io = io,
                                             tol = tol,
                                             constr_viol_tol = constr_viol_tol,
                                             compl_inf_tol = compl_inf_tol,
                                             print_level = print_level_NCL)

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

                    dual_feas = (ncl.meta.ncon != 0) ? norm(grad(ncl.nlp, x_k) - jtprod(ncl.nlp, x_k, y_k) - (z_k_L - z_k_U)[1:nx], Inf) : norm(grad(ncl.nlp, x_k) - (z_k_L - z_k_U)[1:nx], Inf)
                    primal_feas = norm(setdiff(vcat(cons(ncl.nlp, x_k) - ncl.nlp.meta.lcon, ncl.nlp.meta.ucon - cons(ncl.nlp, x_k)), [Inf, -Inf]), Inf)

                    #(print_level_solver == 0) && rm(output_dir_solver)

                    return GenericExecutionStats(status, ncl ;
                                                solution = x_k,
                                                iter = max(iter_count, k),
                                                primal_feas = primal_feas,
                                                dual_feas = dual_feas,
                                                objective = obj(ncl.nlp, x_k),
                                                elapsed_time = 0,
                                                #! doesn't work... counters = nlp.counters,
                                                solver_specific = Dict(:multipliers_con => y_k,
                                                                       :multipliers_L => z_k_L[1:nx],
                                                                       :multipliers_U => z_k_U[1:nx],
                                                                       :internal_msg => converged ? Symbol("Solve_Succeeded") : Symbol("Solve_Failed"),
                                                                       :residuals => r_k
                                                                      )
                                                )
                #else
                #    ncl.ρ = ncl.ρ * τ #! A voir !
                end
            end

        else
   #** II.3 Increase penalization
            ncl.ρ = min(ncl.ρ*τ_ρ, ρ_max) # increase the penalization
            #η_k = η_end / (1 + ncl.ρ ^ α) # Change infeasibility (heuristic)
            if ncl.ρ == ρ_max
                @warn "\nin NCLSolve($(ncl.nlp.meta.name)): maximum penalty ρ = " * string(ρ_max) * " reached at iteration k = " * string(k)
            end
        end
        # ? Chez Nocedal & Wright, p.521, on a : ω_k = 1/ncl.ρ, ncl.ρ = 100ρ_k, η_k = 1/ncl.ρ^0.1
    end
end
