export KKTCheck

"""
###############################
mult_format_check documentation
    mult_format_check verifys that z_U and z_L are given in the right format to KKTCheck.

    !!! Important note !!! The convention is :
    (P) min f(x)
        s.t. c(x) ≥ 0

    And then
        multipliers y ≥ 0
        Lagrangien(x, y) = f(x) - y' * c(x)
        ∇_{x}[lag(x, y)] = ∇_{x}[f(x)] - t(Jac_{c(x)}) * y - (z_L - z_U)
###############################
"""
function mult_format_check(z_U::Vector{<:Float64}, z_L::Vector{<:Float64}, ϵ::Float64) #::Tuple{Vector{<:Float64}, Vector{<:Float64}}
    if (any(z_U .< -ϵ) & any(z_U .> ϵ))
        println("    z_U = ", z_U)

        error("sign problem of z_U passed in argument to KKTCheck (detected by mult_format_check function).
               Multipliers are supposed to be ≥ 0.
               Here, some components are negatives")
    end

    if (any(z_L .< -ϵ) & any(z_L .> ϵ))
        println("    z_L = ", z_L)

        error("sign problem of z_L passed in argument to KKTCheck (detected by mult_format_check function).
               Multipliers are supposed to be ≥ 0.
               Here, some components are negatives")
    end

    if all(z_U .< ϵ) & any(z_U .< - ϵ)
        @warn "z_U was ≤ ϵ (complementarity tolerance) and non zero so it was changed to its opposite. Multipliers are supposed to be all ≥ 0"
        z_U = - z_U
    end

    if all(z_L .< ϵ) & any(z_L .< - ϵ)
        @warn "z_L was ≤ ϵ (complementarity tolerance) and non zero so it was changed to its opposite. Multipliers are supposed to be all ≥ 0"
        z_L = - z_L
    end

    return z_U, z_L
end




function KKTCheck(nlp, x, y, z_U, z_L, file::String; kwargs...)
    out = open(file, "w") do io
        KKTCheck(nlp, x, y, z_U, z_L, io; kwargs...)
    end
    return out
end

"""
#######################
KKTCheck Documentation
    KKTCheck tests if (x, y, z_U, z_L) is a solution of the KKT conditions of the nlp problem (nlp follows the NLPModels.jl formalism, it is suposed to be an AbstractNLPModel), within
        ω as a tolerance for the lagrangian gradient norm
        η as a tolerance for constraint infeasibility
        ϵ as a tolerance for complementarity checking
    the print_level parameter control the verbosity of the function : 0 : nothing
                                                                    # 1 : Function call and result
                                                                    # 2 : Further information in case of failure
                                                                    # 3... : Same, increasing information
                                                                    # 6 & 7 : Shows full vectors, not advised if your problem has a big size

    !!! Important note !!! the lagrangian is considered as :
        l(x, y) = f(x) - y' * c(x)
        with c(x) ≥ 0
                y ≥ 0
    And then
        multipliers y ≥ 0
        Lagrangien(x, y) = f(x) - y' * c(x)
        ∇_{x}[lag(x, y)] = ∇_{x}[f(x)] - t(Jac_{c(x)}) * y - (z_L - z_U)

    Another remark: If z_U is not given (empty), we treat in two different ways complementarity. We can check everything as a range bound constraint in this cae, and when z_L and z_U are given separately,
#######################
"""
function KKTCheck(nlp::AbstractNLPModel,                          # Problem considered
                  #* Position and multipliers
                  x::Vector{<:AbstractFloat},                           # Potential solution
                  y::Vector{<:AbstractFloat},                           # Lagrangian multiplier for constraint
                  z_U::Vector{<:AbstractFloat},                         # Lagrangian multiplier for upper bound constraint
                  z_L::Vector{<:AbstractFloat},                         # Lagrangian multiplier for lower bound constraint
                  io::IO=stdout;


                  #* Tolerances
                  tol::Float64 = 1e-6,                            # Tolerance for lagrangian gradient norm
                  constr_viol_tol::Float64 = 1e-4,                # Tolerance or constraint violation
                  compl_inf_tol::Float64 = 1e-4,                  # Tolerance for complementarity
                  acc_factor::Float64 = 100.,

                  #* Print options
                  print_level::Int = 0,                           # Verbosity level : 0 : nothing
                                                                                    # 1 : Function call and result
                                                                                    # 2 : Further information in case of failure
                                                                                    # 3... : Same, increasing information
                                                                                    # 6 & 7 : Shows full vectors, not advised if your problem has a big size
                 ) #::Dict{String, Any} # dictionnary containing booleans optimality and acceptable optimality, and values of feasibility

    #** 0. Initial settings
    #** 0.1 Notations

    ω::Float64 = tol
    acc_ω::Float64 = acc_factor * tol

    η::Float64 = constr_viol_tol
    acc_η::Float64 = acc_factor * constr_viol_tol

    ϵ::Float64 = compl_inf_tol
    acc_ϵ::Float64 = acc_factor * compl_inf_tol

    optimal::Bool = true # default values, we will update them, accross the following tests
    acceptable::Bool = true
    norm_grad_lag::Float64 = -1.
    primal_feas::Float64 = Inf
    dual_feas::Float64 = Inf
    complementarity_feas::Float64 = Inf

    #** 0.2 Print
    if print_level ≥ 1
        @printf(io, "\nKKTCheck called on %s \n", nlp.meta.name)
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
        error("Problem with free constraints at indices " * string(nlp.meta.jfree) * " passed to KKTCheck")
    end
    if nlp.meta.jinf != []
        error("Problem with infeasible constraints at indices " * string(nlp.meta.jinf) * " passed to KKTCheck")
    end
    if nlp.meta.iinf != []
        error("Problem with infeasible bound constraints at indices " * string(nlp.meta.iinf) * " passed to KKTCheck")
    end

    #** I. Fast check
    #** I.1 Computation
    dual_feas = (nlp.meta.ncon != 0) ? norm(grad(nlp, x) - jtprod(nlp, x, y) - z, Inf) : norm(grad(nlp, x) - z, Inf)
    primal_feas = (nlp.meta.ncon != 0) ? norm(setdiff(vcat(cons(nlp, x) - nlp.meta.lcon, nlp.meta.ucon - cons(nlp, x)), [Inf, -Inf]), Inf) : 0.

    compl_bound_low = vcat(setdiff(z .* (x - nlp.meta.lvar), [Inf, -Inf, NaN, -NaN]), 0.) # Just to get rid of infinite values (due to free variables or constraints) and NaN, due to x[i] * uvar[i] = 0 * Inf = NaN
    compl_bound_upp = vcat(setdiff(z .* (x - nlp.meta.uvar), [Inf, -Inf, NaN, -NaN]), 0.) # zeros are added just to avoid empty vectors (easier for comparison after, but has no influence)

    if length(compl_bound_low) < length(compl_bound_upp)
        append!(compl_bound_low, zeros(Float64, length(compl_bound_upp) - length(compl_bound_low)))
    else
        append!(compl_bound_upp, zeros(Float64, length(compl_bound_low) - length(compl_bound_upp)))
    end

    compl_var_low = (nlp.meta.ncon != 0) ? vcat(setdiff(y .* (cons(nlp, x) - nlp.meta.lcon), [Inf, -Inf, NaN, -NaN]), 0.) : [0.]
    compl_var_upp = (nlp.meta.ncon != 0) ? vcat(setdiff(y .* (cons(nlp, x) - nlp.meta.ucon), [Inf, -Inf, NaN, -NaN]), 0.) : [0.]

    if length(compl_var_low) < length(compl_var_upp)
        append!(compl_var_low, zeros(Float64, length(compl_var_upp) - length(compl_var_low)))
    else
        append!(compl_var_upp, zeros(Float64, length(compl_var_low) - length(compl_var_upp)))
    end

    complementarity_feas = norm(vcat(compl_bound_low, compl_bound_upp, compl_var_low, compl_var_upp), Inf)

    if print_level ≤ 0
        #** I.2 Tests
        if dual_feas ≥ ω
            optimal = false
            if dual_feas ≥ acc_ω
                acceptable = false

                KKT_res = Dict("optimal" => optimal,
                               "acceptable" => acceptable,
                               "primal_feas" => primal_feas,
                               "dual_feas" => dual_feas,
                               "complementarity_feas" => complementarity_feas)

                return KKT_res
            end
        end

        if primal_feas ≤ - η
            optimal = false

            if primal_feas ≤ - acc_η
                acceptable = false

                KKT_res = Dict("optimal" => optimal,
                               "acceptable" => acceptable,
                               "primal_feas" => primal_feas,
                               "dual_feas" => dual_feas,
                               "complementarity_feas" => complementarity_feas)

                return KKT_res
            end
        end

        if any(.!(-ϵ .≤ compl_bound_low .≤ ϵ)  .&  .!(-ϵ .≤ compl_bound_upp .≤ ϵ))
            optimal = false

            if any(.!(-acc_ϵ .≤ compl_bound_low .≤ acc_ϵ)  .&  .!(-acc_ϵ .≤ compl_bound_upp .≤ acc_ϵ))
                acceptable = false

                KKT_res = Dict("optimal" => optimal,
                               "acceptable" => acceptable,
                               "primal_feas" => primal_feas,
                               "dual_feas" => dual_feas,
                               "complementarity_feas" => complementarity_feas)

                return KKT_res
            end
        end

        if any(.!(-ϵ .≤ compl_var_low .≤ ϵ)  .&  .!(-ϵ .≤ compl_var_upp .≤ ϵ))
            optimal = false

            if any(.!(-acc_ϵ .≤ compl_var_low .≤ acc_ϵ)  .&  .!(-acc_ϵ .≤ compl_var_upp .≤ acc_ϵ))
                acceptable = false

                KKT_res = Dict("optimal" => optimal,
                               "acceptable" => acceptable,
                               "primal_feas" => primal_feas,
                               "dual_feas" => dual_feas,
                               "complementarity_feas" => complementarity_feas)

                return KKT_res
            end
        end


        KKT_res = Dict("optimal" => optimal,
                       "acceptable" => acceptable,
                       "primal_feas" => primal_feas,
                       "dual_feas" => dual_feas,
                       "complementarity_feas" => complementarity_feas)

        return KKT_res

    else
        #** II. Bounds constraints
        #** II.1 For free variables
        for i in nlp.meta.ifree
            #** II.1.1 Feasibility USELESS (because variables are free)
            #** II.1.2 Complementarity for bounds
            if !(-ϵ ≤ z[i] ≤ ϵ)
                if !(-acc_ϵ ≤ z[i] ≤ acc_ϵ)
                    if print_level ≥ 1
                        if print_level ≥ 2
                            @printf(io, "    Multiplier not acceptable as zero for free variable %d \n", i)

                            if print_level ≥ 3
                                @printf(io, "      z[%d]             = %7.2e\n", i, z[i])
                                @printf(io, "      x[%d]             = %7.2e\n", i, x[i])
                                @printf(io, "      nlp.meta.lvar[%d] = %7.2e\n", i, nlp.meta.lvar[i])
                                @printf(io, "      nlp.meta.uvar[%d] = %7.2e\n", i, nlp.meta.uvar[i])
                            end
                        end
                        write(io, "\n  ------- Not solved to acceptable level for KKT conditions ----------\n")

                    end
                    acceptable = false

                else
                    if print_level ≥ 1
                        if print_level ≥ 2
                            @printf(io, "    Multiplier acceptable as zero for free variable %d, but suboptimal \n", i)

                            if print_level ≥ 3
                                @printf(io, "      z[%d]             = %7.2e\n", i, z[i])
                                @printf(io, "      x[%d]             = %7.2e\n", i, x[i])
                                @printf(io, "      nlp.meta.lvar[%d] = %7.2e\n", i, nlp.meta.lvar[i])
                                @printf(io, "      nlp.meta.uvar[%d] = %7.2e\n", i, nlp.meta.uvar[i])
                            end
                        end
                        write(io, "\n  ------- Not optimally fitting with KKT conditions (but acceptable) ----------\n")
                    end
                end

                optimal = false
            end
        end

        #** II.2 Bounded variables
        for i in setdiff([i for i in 1:nlp.meta.nvar], nlp.meta.ifree) #free variables were treated brefore
            #** II.2.1 Feasibility
            if !(nlp.meta.lvar[i] - η ≤ x[i] ≤ nlp.meta.uvar[i] + η)
                if !(nlp.meta.lvar[i] - acc_η ≤ x[i] ≤ nlp.meta.uvar[i] + acc_η)
                    if print_level ≥ 1
                        if print_level ≥ 2
                            @printf(io, "    variable %d out of bounds + acceptable tolerance\n", i)

                            if print_level ≥ 3
                                @printf(io, "      x[%d] = %7.2e\n", i, x[i])
                                @printf(io, "      nlp.meta.lvar[%d] = %7.2e\n", i, nlp.meta.lvar[i])
                                @printf(io, "      nlp.meta.uvar[%d] = %7.2e\n", i, nlp.meta.uvar[i])
                            end
                        end
                        write(io, "\n  ------- Not solved to acceptable level for KKT conditions ----------\n")

                    end
                    acceptable = false

                else
                    if print_level ≥ 1
                        if print_level ≥ 2
                            @printf(io, "    variable %d out of bounds + optimal tolerance\n", i)

                            if print_level ≥ 3
                                @printf(io, "      x[%d] = %7.2e\n", i, x[i])
                                @printf(io, "      nlp.meta.lvar[%d] = %7.2e\n", i, nlp.meta.lvar[i])
                                @printf(io, "      nlp.meta.uvar[%d] = %7.2e\n", i, nlp.meta.uvar[i])
                            end
                        end
                        write(io, "\n  ------- Not optimally fitting with KKT conditions (but acceptable) ----------\n")
                    end
                end

                optimal = false
            end

            #** II.2.2 Complementarity for bounds
            if !( (-ϵ ≤ z[i] * (x[i] - nlp.meta.lvar[i]) ≤ ϵ)  |  (-ϵ ≤ z[i] * (x[i] - nlp.meta.uvar[i]) ≤ ϵ) ) # Complementarity condition
                if !( (-acc_ϵ ≤ z[i] * (x[i] - nlp.meta.lvar[i]) ≤ acc_ϵ)  |  (-acc_ϵ ≤ z[i] * (x[i] - nlp.meta.uvar[i]) ≤ acc_ϵ) ) # Complementarity acceptable condition
                    if print_level ≥ 1
                        if print_level ≥ 2
                            @printf(io, "    both complementarities = %7.2e or %7.2e are out of acceptable tolerance acc_ϵ = %7.2e. See bound var %d\n", z[i] * (x[i] - nlp.meta.lvar[i]), z[i] * (x[i] - nlp.meta.uvar[i]), acc_ϵ, i)

                            if print_level ≥ 3
                                @printf(io, "      z[%d]             = %7.2e \n", i, z[i])
                                @printf(io, "      x[%d]             = %7.2e \n", i, x[i])
                                @printf(io, "      nlp.meta.lvar[%d] = %7.2e \n", i, nlp.meta.lvar[i])
                                @printf(io, "      nlp.meta.uvar[%d] = %7.2e \n", i, nlp.meta.uvar[i])
                            end
                        end
                        write(io, "\n  ------- Not solved to acceptable level for KKT conditions ----------\n")

                    end

                    acceptable = false
                else
                    if print_level ≥ 1
                        if print_level ≥ 2
                            @printf(io, "    one of the complementarities = %7.2e or %7.2e is out of tolerance ϵ = %7.2e (but still acceptable). See bound var %d\n", z[i] * (x[i] - nlp.meta.lvar[i]), z[i] * (x[i] - nlp.meta.uvar[i]), ϵ, i)

                            if print_level ≥ 3
                                @printf(io, "      z[%d]             = %7.2e \n", i, z[i])
                                @printf(io, "      x[%d]             = %7.2e \n", i, x[i])
                                @printf(io, "      nlp.meta.lvar[%d] = %7.2e \n", i, nlp.meta.lvar[i])
                                @printf(io, "      nlp.meta.uvar[%d] = %7.2e \n", i, nlp.meta.uvar[i])
                            end
                        end
                        write(io, "\n  ------- Not optimally fitting with KKT conditions (but acceptable) ----------\n")

                    end
                end

                optimal = false
            end
        end

        #** III. Other constraints
        #** III.0 Precomputation
        c_x = cons(nlp, x) # Precomputation

        #** III.1 Feasibility
        for i in 1:nlp.meta.ncon
            if !(nlp.meta.lcon[i] - η ≤ c_x[i] ≤ nlp.meta.ucon[i] + η)
                if !(nlp.meta.lcon[i] - acc_η ≤ c_x[i] ≤ nlp.meta.ucon[i] + acc_η)
                    if print_level ≥ 1
                        if print_level ≥ 2
                            @printf(io, "    constraint %d out of bounds + acceptable tolerance\n", i)

                            if print_level ≥ 3
                                @printf(io, "      c_x[%d]               = %7.2e \n", i, c_x[i])
                                @printf(io, "      nlp.meta.ucon[%d] + acc_η = %7.2e \n", i, nlp.meta.ucon[i] + acc_η)
                                @printf(io, "      nlp.meta.lcon[%d] - acc_η = %7.2e \n", i, nlp.meta.lcon[i] - acc_η)
                            end
                        end
                        write(io, "\n  ------- Not solved to acceptable level for KKT conditions ----------\n")

                    end
                    acceptable = false

                else
                    if print_level ≥ 1
                        if print_level ≥ 2
                            @printf(io, "    constraint %d out of bounds + tolerance\n", i)

                            if print_level ≥ 3
                                @printf(io, "      c_x[%d]               = %7.2e \n", i, c_x[i])
                                @printf(io, "      nlp.meta.ucon[%d] + η = %7.2e \n", i, nlp.meta.ucon[i] + η)
                                @printf(io, "      nlp.meta.lcon[%d] - η = %7.2e \n", i, nlp.meta.lcon[i] - η)
                            end
                        end
                        write(io, "\n  ------- Not optimally fitting with KKT conditions (but acceptable) ----------\n")

                    end
                end

                optimal = false
            end
        end

        #** III.2 Complementarity
        for i in 1:nlp.meta.ncon # upper constraints
            if !( (-ϵ ≤ (y[i] * (c_x[i] - nlp.meta.ucon[i])) ≤ ϵ)  |  (-ϵ ≤ (y[i] * (c_x[i] - nlp.meta.lcon[i])) ≤ ϵ) )  # Complementarity condition (for range constraint, we have necessarily : [y[i] * (c_x[i] - nlp.meta.lcon[i])] * [y[i] * (c_x[i] - nlp.meta.ucon[i])] = 0
                if !( (-acc_ϵ ≤ (y[i] * (c_x[i] - nlp.meta.ucon[i])) ≤ acc_ϵ)  |  (-acc_ϵ ≤ (y[i] * (c_x[i] - nlp.meta.lcon[i])) ≤ acc_ϵ) )  # Complementarity condition (for range constraint, we have necessarily : [y[i] * (c_x[i] - nlp.meta.lcon[i])] * [y[i] * (c_x[i] - nlp.meta.ucon[i])] = 0
                    if print_level ≥ 1
                        if print_level ≥ 2
                            @printf(io, "    one of the two complementarities %7.2e or %7.2e is out of acceptable tolerance acc_ϵ = %7.2e. See cons %d \n", y[i] * (c_x[i] - nlp.meta.ucon[i]), (y[i] * (c_x[i] - nlp.meta.lcon[i])), acc_ϵ, i)

                            if print_level ≥ 3
                                @printf(io, "      y[%d]             = %7.2e \n", i, y[i])
                                @printf(io, "      c_x[%d]           = %7.2e \n", i, c_x[i])
                                @printf(io, "      nlp.meta.ucon[%d] = %7.2e \n", i, nlp.meta.ucon[i])
                                @printf(io, "      nlp.meta.lcon[%d] = %7.2e \n", i, nlp.meta.lcon[i])
                            end
                        end

                        write(io, "\n  ------- Not solved to acceptable level for KKT conditions ----------\n")
                    end
                    acceptable = false

                else
                    if print_level ≥ 1
                        if print_level ≥ 2
                            @printf(io, "    one of the two complementarities %7.2e or %7.2e is out of tolerance ϵ = %7.2e. See cons %d \n", y[i] * (c_x[i] - nlp.meta.ucon[i]), (y[i] * (c_x[i] - nlp.meta.lcon[i])), ϵ, i)

                            if print_level ≥ 3
                                @printf(io, "      y[%d]             = %7.2e \n", i, y[i])
                                @printf(io, "      c_x[%d]           = %7.2e \n", i, c_x[i])
                                @printf(io, "      nlp.meta.ucon[%d] = %7.2e \n", i, nlp.meta.ucon[i])
                                @printf(io, "      nlp.meta.lcon[%d] = %7.2e \n", i, nlp.meta.lcon[i])
                            end
                        end

                        write(io, "\n  ------- Not optimally fitting with KKT conditions (but acceptable) ----------\n")
                    end
                end

                optimal = false
            end
        end

        #** IV. Lagrangian
        #** IV.1 Computation
        ∇f_x = grad(nlp, x)
        if nlp.meta.ncon != 0 # just to avoid DimensionMismatch with ∇f_x - [].
            ∇lag_x = ∇f_x - jtprod(nlp, x, y) - z
        else
            ∇lag_x = ∇f_x - z
        end
        norm_grad_lag = norm(∇lag_x, Inf)

        #** IV.2 Test & print
        if norm_grad_lag > ω # Not a stationnary point for the lagrangian
            if norm_grad_lag > acc_ω # Not an acceptable stationnary point for the lagrangian
                if print_level ≥ 1
                    if print_level ≥ 2
                        @printf(io, "    Lagrangian gradient norm = %7.2e is greater than acceptable tolerance acc_ω = %7.2e \n", norm_grad_lag, acc_ω)

                        if print_level ≥ 4
                            if nlp.meta.ncon != 0
                                @printf(io, "      ‖∇f_x‖                = %7.2e \n", norm(∇f_x, Inf))
                                @printf(io, "      ‖∇f_x - t(Jac_x) * y‖ = %7.2e \n", norm(∇f_x - jtprod(nlp, x, y), Inf))
                                @printf(io, "      ‖z‖                   = %7.2e \n", norm(z, Inf))
                                @printf(io, "      ‖∇lag_x‖              = %7.2e \n", norm_grad_lag)
                            else
                                @printf(io, "      ‖∇f_x‖   = %7.2e \n", norm(∇f_x, Inf))
                                @printf(io, "      ‖- z‖    = %7.2e \n", norm(z, Inf))
                                @printf(io, "      ‖∇lag_x‖ = %7.2e \n", norm_grad_lag)
                            end
                        end
                    end

                    write(io, "\n  ------- Not solved to acceptable level for KKT conditions ----------\n")
                end

                acceptable = false
            else
                if print_level ≥ 1
                    if print_level ≥ 2
                        @printf(io, "    Lagrangian gradient norm = %7.2e is greater than tolerance ω = %7.2e \n", norm_grad_lag, ω)

                        if print_level ≥ 7
                            if nlp.meta.ncon != 0
                                @printf(io, "      ‖∇f_x‖                = %7.2e \n", norm(∇f_x, Inf))
                                @printf(io, "      ‖∇f_x - t(Jac_x) * y‖ = %7.2e \n", norm(∇f_x - jtprod(nlp, x, y), Inf))
                                @printf(io, "      ‖z‖                   = %7.2e \n", norm(z, Inf))
                                @printf(io, "      ‖∇lag_x‖              = %7.2e \n", norm_grad_lag)
                            else
                                @printf(io, "      ‖∇f_x‖   = %7.2e \n", norm(∇f_x, Inf))
                                @printf(io, "      ‖- z‖    = %7.2e \n", norm(z, Inf))
                                @printf(io, "      ‖∇lag_x‖ = %7.2e \n", norm_grad_lag)
                            end
                        end
                    end

                    write(io, "\n  ------- Not optimally fitting with KKT conditions (but acceptable) ----------\n")
                end
            end

            optimal = false
        end

        #** V Returns dictionnary
        if print_level ≥ 1
            if optimal
                @printf(io, "\n    %s problem solved to optimal level !\n", nlp.meta.name)
            elseif acceptable
                @printf(io, "\n    %s problem solved to acceptable level\n", nlp.meta.name)
            end
        end

        KKT_res = Dict("optimal" => optimal,
                       "acceptable" => acceptable,
                       "primal_feas" => primal_feas,
                       "dual_feas" => dual_feas,
                       "complementarity_feas" => complementarity_feas)

        return KKT_res
    end
end
