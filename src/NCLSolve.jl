
export NCLSolve

# NCLSolve called with a string outputs to file
function NCLSolve(nlp::AbstractNLPModel, file::String; kwargs...)
    out = open(file, "w") do io
        NCLSolve(NCLModel(nlp), io; kwargs...)
    end

    return out
end

#NCLSolve called with an NLPModel
function NCLSolve(nlp::AbstractNLPModel ; kwargs...)
  return NCLSolve(NCLModel(nlp) ; kwargs...)
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
                                    :multipliers_L => z_L_k,    #                              upper bounds
                                    :multipliers_U => z_U_k     #                              lower bounds
                                    )
                )
#################################
"""
function NCLSolve(ncl::NCLModel,                             # Problem to be solved by this method
                  io::IO = stdout;                           # where to print iterations

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
                  scale_mu_init::Float64 = 0.3,

                  init_penal::Float64 = 10.0,
                  init_tol::Float64 = 0.1,
                  init_constr_viol_tol::Float64 = 0.1,
                  init_compl_inf_tol::Float64 = 0.1,

                  max_iter_NCL::Int = 20,                  # Maximum number of iterations for the NCL method
                  KKT_checking::Bool = false,                # To choose if you want to check KKT conditions in the loop, or just stop when residuals are small enough.

                  #* Options for solver
                  max_iter_solver::Int = 1000,               # Maximum number of iterations for the subproblem solver
                  print_level_solver::Int = 0,               # Options for printing iterations of the subproblem solver

                  #* Options of NCL print
                  print_level_NCL::Int = 0,                  # Options for printing iterations of the NCL method : 0, nothing;
                                                                                                                    # 1, calls to functions and conclusion;
                                                                                                                    # 2, calls, little informations on iterations;
                                                                                                                    # 3, calls, more information about iterations (and erors in KKTCheck);
                                                                                                                    # 4, calls, KKTCheck, iterations, little information from the solver;                                                                                                                         # and so on until 7 (no further details)

                 ) #::GenericExecutionStats                   # See NLPModelsIpopt / NLPModelsKnitro and SolverTools for further details on this structure

    #** I.0 Solution with NCL
    nr = ncl.nr
    nx = ncl.nx

    #** I. Names and variables
    #** I.1 Constants & scale parameters
    mu_init = 1.0e-3

    τ_ρ = scale_penal # scale (used to update the ρ_k penalization)
    τ_ϵ = scale_compl_inf_tol
    τ_η = scale_constr_viol_tol
    τ_ω = scale_tol
    τ_mu_init = scale_mu_init

    α = 0.1 # Constant (α needs to be < 1)
    β = 0.2 # Constant

    #** I.2 Parameters
    ncl.ρ = 100.0 # penalization
    ρ_max = max_penal # biggest penalization authorized

    ω_end = tol #global tolerance, in argument
    ω_k = init_tol # sub problem initial tolerance
    ω_min = min_tol

    η_end = 0.5 * constr_viol_tol #global infeasibility in argument
    η_k = init_constr_viol_tol # sub problem infeasibility
    η_min = min_constr_viol_tol # smallest infeasibility authorized

    ϵ_end = compl_inf_tol - η_end
    if ϵ_end < 0
      ϵ_end = compl_inf_tol
      @warn "NCLSolve($(ncl.nlp.meta.name)): your tolerance compl_inf_tol is to low regarding to constr_viol_tol. \nYou should have compl_inf_tol >= constr_viol_tol * 0.5 if you want the solution to satisfy optimal KKT conditions with your level of tolerances."
    end
    ϵ_k = init_compl_inf_tol
    ϵ_min = min_compl_inf_tol


    if nr == 0
      ω_k = tol
      η_k = constr_viol_tol
      ϵ_k = compl_inf_tol
    end

    acc_count = 0
    iter_count = 0
    warn_count = 0

    #** I.3 Solver acceptable parameters
    acceptable_tol::Float64 = acc_factor * ω_k
    acceptable_constr_viol_tol::Float64 = acc_factor * η_k
    acceptable_compl_inf_tol::Float64 = acc_factor * ϵ_k

    #** I.4 Initial variables
    xr_k = ncl.meta.x0
    best_xr_k = xr_k # best solution found, to be passed to solver

    x_k = xr_k[1 : nx]
    r_k = xr_k[nx + 1 : nx + nr]

    norm_r_k_inf = norm(r_k, Inf) #Pre-computation
    best_rNorm = Inf

    y_k = copy(ncl.meta.y0)
    best_y_k = y_k
    z_U_k = zeros(Float64, length(ncl.meta.uvar))
    best_z_U_k = z_U_k
    z_L_k = zeros(Float64, length(ncl.meta.lvar))
    best_z_L_k = z_L_k

    #** I.5 Initial print
    print_level_NCL >= 1 &&	@info @sprintf("\nNCL resolution of %s", ncl.nlp.meta.name)
    print_level_NCL >= 2 &&	println(ncl)
    print_level_NCL >= 1 &&	@info @sprintf("NCLSolve(%s) iterations :", ncl.nlp.meta.name)
    print_level_NCL >= 1 &&	@info @sprintf("%4s  %12s  %34s  %7s  %7s  %7s  %7s  %8s  %9s  %9s  %7s  %7s",
                                          "Iter", "Iter_solver", "Success ?", "‖rₖ‖∞", "ηₖ", "ωₖ", "ρₖ", "μ init", "NCL obj", "NLP obj", "‖yₖ‖", "‖xₖ‖")


    #** II. Optimization loop and return
    k = 0
    converged = false
    solved = false

    local solve_k
    NCL_exit = solver_exit = :Default_Exit_Message #For exit message

    while (k ≤ max_iter_NCL) & !converged
      #** II.0 Iteration counter and mu_init
      k += 1

      if (k%2 == 0) && (k <= 12)
        mu_init /= 10
      elseif (k>=12) && (warn_count == 4)
        print_level_NCL >= 1 &&	@info @sprintf("\nin NCLSolve(%s): reached all limit tolerance, no more evolution possible, failure.", ncl.nlp.meta.name)
        break
      end

      #** II.1 Get subproblem's solution
      #** II.1.1 Solver
      solve_k = NLPModelsIpopt.ipopt(ncl;
                                    x0 = best_xr_k,
                                    y0 = best_y_k,
                                    zL = best_z_L_k,
                                    zU = best_z_U_k,
                                    tol = ω_k,
                                    constr_viol_tol = η_k,
                                    compl_inf_tol = ϵ_k,
                                    acceptable_tol = acceptable_tol,
                                    acceptable_constr_viol_tol = acceptable_constr_viol_tol,
                                    acceptable_compl_inf_tol = acceptable_compl_inf_tol,
                                    print_level = print_level_solver,
                                    warm_start_init_point = "yes",
                                    mu_init = mu_init,
                                    mumps_mem_percent = 5,
                                    dual_inf_tol = 1e-6,
                                    max_iter = max_iter_solver)

      # Get variables
      xr_k = solve_k.solution #pre-access
      x_k .= xr_k[1:nx]
      r_k .= xr_k[nx+1 : nx+nr]
      norm_r_k_inf = norm(r_k, Inf) # update

      # Get multipliers
      #! Warning, ipopt doesn't use our convention in KKTCheck for constraint multipliers, so we took the opposite. For bound multiplier it seems to work though.
      y_k = - solve_k.solver_specific[:multipliers_con]
      z_U_k = solve_k.solver_specific[:multipliers_U]
      z_L_k = solve_k.solver_specific[:multipliers_L]

      iter_count += solve_k.iter

      # Solver's solution
      solver_exit = solve_k.solver_specific[:internal_msg]
      iter_count += solve_k.iter
      solved::Bool = ((solver_exit == :Solve_Succeeded) | (solver_exit == :Solved_To_Acceptable_Level))

      if (norm_r_k_inf <= best_rNorm) & solved #eventually update best point
        best_xr_k .= xr_k
        best_y_k .= y_k
        best_z_U_k .= z_U_k
        best_z_L_k .= z_L_k
        best_rNorm = norm_r_k_inf
      end




      #** II.1.2 Output print
      print_level_NCL >= 1 &&	((objnlp, objncl) = objnlp_objncl(ncl, xr_k))
      print_level_NCL >= 1 &&	@info @sprintf("%4d  %12d  %34s  %7.1e  %7.1e  %7.1e  %7.1e   %7.1e  %9.2e  %9.2e  %7.1e  %7.1e",
                                                k, solve_k.iter, solver_exit, norm_r_k_inf, η_k, ω_k, ncl.ρ, mu_init, objncl, objnlp, norm(ncl.y, Inf), norm(x_k))

      #** II.1.3 Checks about the solver's resolution
      if solver_exit == :Restoration_Failed
        NCL_exit = :Loop_Due_To_Restauration_Fail
        if solve_k.iter <= 1 # because it will return its arguments, and create a loop...
          break
        end
      end
      if solver_exit == :Infeasible_Problem_Detected
        NCL_exit = :Infeasible_Problem_Detected
        break
      end


      #** II.2 Treatment & update
      warn_count = 0
      
      if (norm_r_k_inf ≤ max(η_k, η_end)) # The residual has decreased enough
        #** II.2.1 Update
        ncl.y .+= ncl.ρ * r_k # Updating multiplier

        η_k *= τ_η
        if η_k == η_min
          @warn "\nin NCLSolve($(ncl.nlp.meta.name)): minimum constraint violation η_min = " * string(η_min) * " reached at iteration k = " * string(k)
          warn_count += 1
        end

        ω_k *= τ_ω
        if ω_k == ω_min
          @warn "\nin NCLSolve($(ncl.nlp.meta.name)): minimum tolerance ω_min = " * string(ω_min) * " reached at iteration k = " * string(k)
          warn_count += 1
        end

        ϵ_k *= τ_ϵ
        if ϵ_k == ϵ_min
          @warn "\nin NCLSolve($(ncl.nlp.meta.name)): minimum complementarity infeasibility ϵ_min = " * string(ϵ_min) * " reached at iteration k = " * string(k)
          warn_count += 1
        end

        converged = (best_rNorm == norm_r_k_inf) & (best_rNorm ≤ η_end) & solved #We must have resolved the last NCL problem to consider we solved the NLP

        if converged & KKT_checking #if user additionnaly asks for KKT checking
          D_solved = KKTCheck(ncl.nlp,
                              x_k,
                              y_k,
                              z_U_k[1:nx],
                              z_L_k[1:nx] ;
                              io = io,
                              tol = tol,
                              constr_viol_tol = constr_viol_tol,
                              compl_inf_tol = compl_inf_tol,
                              print_level = print_level_NCL)

          converged = D_solved[:optimal]

          if D_solved[:acceptable]
            acc_count += 1 # if we are still on an acceptable level
          else
            acc_count = 0 # if not, then go back to 0
          end

          if acc_count >= 3 # 3 acceptable for KKT and small residuals, okay, we stop
            converged = true
          end
        end

      else
        #** II.3 Increase penalization
        ncl.ρ = min(ncl.ρ*τ_ρ, ρ_max) # increase the penalization
        if ncl.ρ == ρ_max
          @warn "\nin NCLSolve($(ncl.nlp.meta.name)): maximum penalty ρ = " * string(ρ_max) * " reached at iteration k = " * string(k)
          warn_count += 1
        end
      end
      # ? Chez Nocedal & Wright, p.521, on a : ω_k = 1/ncl.ρ, ncl.ρ = 100ρ_k, η_k = 1/ncl.ρ^0.1
    end

  # Access to returned values
  status = solve_k.status
  dual_feas = solve_k.dual_feas
  primal_feas = ncl.nlp.meta.ncon == 0 ? 0.0 : maximum(max.(0, vcat(ncl.nlp.meta.lcon - cons(ncl.nlp, x_k), cons(ncl.nlp, x_k) - ncl.nlp.meta.ucon)))
  zL = solve_k.solver_specific[:multipliers_L][1:nx]
	zU = solve_k.solver_specific[:multipliers_U][1:nx]
	
  # Determination of the exit message
  if NCL_exit == :Default_Exit_Message
    NCL_exit = :Solve_Succeeded
    if (k >= max_iter_NCL) & solved
      NCL_exit = :Maximum_Iterations_Exceeded
    elseif !converged
      NCL_exit = :Solve_Failed
    end
  end


	print_level_NCL < 1 || @info @sprintf("NCLSolve(%s) EXIT: %s", ncl.nlp.meta.name, string(NCL_exit))

  return GenericExecutionStats(status, ncl,
                                solution = x_k,
                                iter = iter_count,
                                primal_feas = primal_feas,
                                dual_feas = dual_feas,
                                objective = obj(ncl.nlp, x_k),
                                elapsed_time = 0,
                                #! doesn't work... counters = nlp.counters,
                                solver_specific = Dict(:multipliers_con => ncl.y,
                                                      :multipliers_L => zL,
                                                      :multipliers_U => zU,
                                                      :internal_msg => NCL_exit,
                                                      :residuals => r_k
                                                    )
                              )


  

end
