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

#using NLPModelsKnitro

#! TODO Fix closing file problem...

######### TODO #########
######### TODO #########
######### TODO #########

    # TODO (feature)   : Créer un vrai statut
    # TODO KKT_check output in file to fix

    # TODO (recherche) : choix des mu_init à améliorer...
    # TODO (recherche) : Points intérieurs à chaud...
    # TODO (recherche) : tester la proximité des multiplicateurs λ_k de renvoyés par le solveur et le ncl.y du problème (si r petit, probablement proches.)
    # TODO (recherche) : Mieux choisir le pas pour avoir une meilleure convergence
    # TODO (recherche) : update ω_k
    # TODO (recherche) : ajuster eta_end

    # TODO (Plus tard) : Pierric pour choix de alpha, beta, tau...
########## TODO ########
########## TODO ########
########## TODO ########

"""
###############################
mult_format_check documentation
    mult_format_check verifys that z_U and z_L are given in the right format to KKT_check.

    !!! Important note !!! The convention is :
    (P) min f(x)
        s.t. c(x) ≥ 0

    And then
        multipliers λ ≥ 0
        Lagrangien(x, λ) = f(x) - λ' * c(x)
        ∇_{x}[lag(x, λ)] = ∇_{x}[f(x)] - t(Jac_{c(x)}) * λ - (z_L - z_U)

###############################
"""
function mult_format_check(z_U::Vector{<:Float64}, z_L::Vector{<:Float64}, ϵ::Float64) ::Tuple{Vector{<:Float64}, Vector{<:Float64}}
    if (any(z_U .< -ϵ) & any(z_U .> ϵ))
        println("    z_U = ", z_U)

        error("sign problem of z_U passed in argument to KKT_check (detected by mult_format_check function).
               Multipliers are supposed to be ≥ 0.
               Here, some components are negatives")
    end

    if (any(z_L .< -ϵ) & any(z_L .> ϵ))
        println("    z_L = ", z_L)

        error("sign problem of z_L passed in argument to KKT_check (detected by mult_format_check function).
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

####### TODO KKT fast check

function KKT_check(nlp, x, λ, z_U, z_L, file::String; kwargs...)
    out = open(file, "w") do io
        KKT_check(nlp, x, λ, z_U, z_L, io; kwargs...)
    end
    return out
end

"""
#######################
KKT_check Documentation
    KKT_check tests if (x, λ, z_U, z_L) is a solution of the KKT conditions of the nlp problem (nlp follows the NLPModels.jl formalism, it is suposed to be an AbstractNLPModel), within
        ω as a tolerance for the lagrangian gradient norm
        η as a tolerance for constraint infeasibility
        ϵ as a tolerance for complementarity checking
    the print_level parameter control the verbosity of the function : 0 : nothing
                                                                    # 1 : Function call and result
                                                                    # 2 : Further information in case of failure
                                                                    # 3... : Same, increasing information
                                                                    # 6 & 7 : Shows full vectors, not advised if your problem has a big size

    !!! Important note !!! the lagrangian is considered as :
        l(x, λ) = f(x) - λ' * c(x)
        with c(x) ≥ 0
                λ ≥ 0
    And then
        multipliers λ ≥ 0
        Lagrangien(x, λ) = f(x) - λ' * c(x)
        ∇_{x}[lag(x, λ)] = ∇_{x}[f(x)] - t(Jac_{c(x)}) * λ - (z_L - z_U)

    Another remark: If z_U is not given (empty), we treat in two different ways complementarity. We can check everything as a range bound constraint in this cae, and when z_L and z_U are given separately,
#######################
"""
function KKT_check(nlp::AbstractNLPModel,                          # Problem considered

                   #* Position and multipliers
                   x::Vector{<:Float64},                           # Potential solution
                   λ::Vector{<:Float64},                           # Lagrangian multiplier for constraint
                   z_U::Vector{<:Float64},                         # Lagrangian multiplier for upper bound constraint
                   z_L::Vector{<:Float64},                         # Lagrangian multiplier for lower bound constraint

                   io::IO=stdout;

                   #* Tolerances
                   tol::Float64 = 1e-6,                            # Tolerance for lagrangian gradient norm
                   constr_viol_tol::Float64 = 1e-4,                # Tolerance or constraint violation
                   compl_inf_tol::Float64 = 1e-4,                  # Tolerance for complementarity
                   acc_factor::Float64 = 100.,

                   #* Print options
                   print_level::Int = 0,                         # Verbosity level : 0 : nothing
                                                                                     # 1 : Function call and result
                                                                                     # 2 : Further information in case of failure
                                                                                     # 3... : Same, increasing information
                                                                                     # 6 & 7 : Shows full vectors, not advised if your problem has a big size
                  ) ::Dict{String, Any}                                       # true returned if the problem is solved at x, with tolerances specified. false instead.

    acceptable_tol::Float64 = acc_factor * tol
    acceptable_constr_viol_tol::Float64 = acc_factor * constr_viol_tol
    acceptable_compl_inf_tol::Float64 = acc_factor * compl_inf_tol

    #** 0. Initial settings
    #** 0.1 Notations
    ω::Float64 = tol
    acc_ω::Float64 = acceptable_tol

    η::Float64 = constr_viol_tol
    acc_η::Float64 = acceptable_constr_viol_tol

    ϵ::Float64 = compl_inf_tol
    acc_ϵ::Float64 = acceptable_compl_inf_tol

    optimal::Bool = true # default values, we will update them, accross the following tests
    acceptable::Bool = true
    norm_grad_lag::Float64 = -1.
    primal_feas::Float64 = Inf
    dual_feas::Float64 = Inf
    complementarity_feas::Float64 = Inf

    #** 0.2 Print
    if print_level ≥ 1
        @printf(io, "\nKKT_check called on %s \n", nlp.meta.name)
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
        error("Problem with infeasible constraints at indices " * string(nlp.meta.jinf) * " passed to KKT_check")
    end
    if nlp.meta.iinf != []
        error("Problem with infeasible bound constraints at indices " * string(nlp.meta.iinf) * " passed to KKT_check")
    end

    #** I. Fast check
    #** I.1 Computation
    dual_feas = (nlp.meta.ncon != 0) ? norm(grad(nlp, x) - jtprod(nlp, x, λ) - z, Inf) : norm(grad(nlp, x) - z, Inf)
    primal_feas = (nlp.meta.ncon != 0) ? minimum(vcat(cons(nlp, x) - nlp.meta.lcon, nlp.meta.ucon - cons(nlp, x))) : 0.

    compl_bound_low = vcat(setdiff(z .* (x - nlp.meta.lvar), [Inf, -Inf, NaN]), 0.) # Just to get rid of infinite values (due to free variables or constraints)
    compl_bound_upp = vcat(setdiff(z .* (x - nlp.meta.uvar), [Inf, -Inf, NaN]), 0.) # zeros are added just to avoid empty vectors (easier for comparison after, but has no influence)

    if length(compl_bound_low) < length(compl_bound_upp)
        append!(compl_bound_low, zeros(Float64, length(compl_bound_upp) - length(compl_bound_low)))
    else
        append!(compl_bound_upp, zeros(Float64, length(compl_bound_low) - length(compl_bound_upp)))
    end

    compl_var_low = (nlp.meta.ncon != 0) ? vcat(setdiff(λ .* (cons(nlp, x) - nlp.meta.lcon), [Inf, -Inf]), 0.) : [0.]
    compl_var_upp = (nlp.meta.ncon != 0) ? vcat(setdiff(λ .* (cons(nlp, x) - nlp.meta.ucon), [Inf, -Inf]), 0.) : [0.]

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
            #** II.1.1 Feasability USELESS (because variables are free)
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
            #** II.2.1 Feasability
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

        #** III.1 Feasability
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
            if !( (-ϵ ≤ (λ[i] * (c_x[i] - nlp.meta.ucon[i])) ≤ ϵ)  |  (-ϵ ≤ (λ[i] * (c_x[i] - nlp.meta.lcon[i])) ≤ ϵ) )  # Complementarity condition (for range constraint, we have necessarily : [λ[i] * (c_x[i] - nlp.meta.lcon[i])] * [λ[i] * (c_x[i] - nlp.meta.ucon[i])] = 0
                if !( (-acc_ϵ ≤ (λ[i] * (c_x[i] - nlp.meta.ucon[i])) ≤ acc_ϵ)  |  (-acc_ϵ ≤ (λ[i] * (c_x[i] - nlp.meta.lcon[i])) ≤ acc_ϵ) )  # Complementarity condition (for range constraint, we have necessarily : [λ[i] * (c_x[i] - nlp.meta.lcon[i])] * [λ[i] * (c_x[i] - nlp.meta.ucon[i])] = 0
                    if print_level ≥ 1
                        if print_level ≥ 2
                            @printf(io, "    one of the two complementarities %7.2e or %7.2e is out of acceptable tolerance acc_ϵ = %7.2e. See cons %d \n", λ[i] * (c_x[i] - nlp.meta.ucon[i]), (λ[i] * (c_x[i] - nlp.meta.lcon[i])), acc_ϵ, i)

                            if print_level ≥ 3
                                @printf(io, "      λ[%d]             = %7.2e \n", i, λ[i])
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
                            @printf(io, "    one of the two complementarities %7.2e or %7.2e is out of tolerance ϵ = %7.2e. See cons %d \n", λ[i] * (c_x[i] - nlp.meta.ucon[i]), (λ[i] * (c_x[i] - nlp.meta.lcon[i])), ϵ, i)

                            if print_level ≥ 3
                                @printf(io, "      λ[%d]             = %7.2e \n", i, λ[i])
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
            ∇lag_x = ∇f_x - jtprod(nlp, x, λ) - z
        else
            ∇lag_x = ∇f_x - z
        end
        norm_grad_lag = norm(∇lag_x, Inf)

        #** IV.2 Test & print
        if norm_grad_lag > ω # Not a stationnary point for the lagrangian
            if norm_grad_lag > acc_ω # Not an acceptable stationnary point for the lagrangian
                if print_level ≥ 1
                    if print_level ≥ 2
                        @printf(io, "    Lagrangian gradient norm = %7.2e is greater acceptable than tolerance acc_ω = %7.2e \n", norm_grad_lag, acc_ω)

                        if print_level ≥ 4
                            if nlp.meta.ncon != 0
                                @printf(io, "      ‖∇f_x‖                = %7.2e \n", norm(∇f_x, Inf))
                                @printf(io, "      ‖∇f_x - t(Jac_x) * λ‖ = %7.2e \n", norm(∇f_x - jtprod(nlp, x, λ), Inf))
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
                                @printf(io, "      ‖∇f_x - t(Jac_x) * λ‖ = %7.2e \n", norm(∇f_x - jtprod(nlp, x, λ), Inf))
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

# NCLSolve called with a string outputs to file
function NCLSolve(nlp, file::String; kwargs...)
    out = open(file, "w") do io
        NCLSolve(nlp, io; kwargs...)
    end
    return out
end

# NCLSolve called with a generic model first constructs an NCLModel
function NCLSolve(nlp::AbstractNLPModel, args...; linear_residuals::Bool = true, kwargs...)
    ncl = NCLModel(nlp;
                   res_val_init = 0.,
                   res_lin_cons = linear_residuals)
    NCLSolve(ncl, args...; kwargs...)
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
                solver_specific=Dict(:multipliers_con => λ_k,   # lagrangian multipliers for : constraints
                                    :multipliers_L => z_k_L,    #                              upper bounds
                                    :multipliers_U => z_k_U     #                              lower bounds
                                    )
                )
    Returns (x (solution), y (lagrangian multipliers for constraints), z (lagrangian multpliers for bound constraints))
#################################
"""
function NCLSolve(ncl::NCLModel,                                                     # Problem to be solved by this method (see NCLModel.jl for further details)
                  io::IO=stdout;
                  #* Optimization parameters
                  tol::Float64 = 1e-6,                                                # Tolerance for the gradient lagrangian norm
                  constr_viol_tol::Float64 = 1e-6,                                     # Tolerance for the infeasibility accepted for ncl
                  compl_inf_tol::Float64 = 0.0001,                                     # Tolerance for the complementarity accepted for ncl
                  acc_factor::Float64 = 100.,

                  #* Options for NCL
                  max_penal::Float64 = 1e15,                                           # Maximal penalization authorized (ρ_max)
                  min_infeas::Float64 = 1e-10,                                          # Minimal infeasibility authorized (η_min)
                  max_iter_NCL::Int = 20,                                            # Maximum number of iterations for the NCL method
                  print_level_NCL::Int = 0,                                          # Options for printing iterations of the NCL method : 0, nothing;
                                                                                                                                           # 1, calls to functions and conclusion;
                                                                                                                                           # 2, calls, little informations on iterations;
                                                                                                                                           # 3, calls, more information about iterations (and erors in KKT_check);
                                                                                                                                           # 4, calls, KKT_check, iterations, little information from the solver;                                                                                                                         # and so on until 7 (no further details)
                  # output_file_print_NCL::Bool = false,                                 # Choose if you want NCL iterations in a separate file
                  # output_file_name_NCL::String = "NCL.log",                            # The name of the separate file
                  # output_file_NCL::IOStream = open("NCL.log", write=true),             # The file itself, openned, if you like. Only if you want to include it in another file. Won't be closed at the end
                  KKT_checking::Bool = false,                                           # To choose if you want to check KKT conditions in the loop, or just stop when residuals are small enough.

                  #* Options for solver
                  max_iter_solver::Int = 1000,                                       # Maximum number of iterations for the subproblem solver
                  print_level_solver::Int = 0,                                       # Options for printing iterations of the subproblem solver
                  output_file_print_solver::Bool = false,                              # Choose if you want subproblems iterations in a separate file
                  warm_start_init_point::Bool = true,                               # "yes" to choose warm start in the subproblem solving. "no" for normal solving.

                 ) ::GenericExecutionStats                                              # See NLPModelsIpopt / NLPModelsKnitro and SolverTools for further details on this structure

    acceptable_tol::Float64 = acc_factor * tol
    acceptable_constr_viol_tol::Float64 = acc_factor * constr_viol_tol
    acceptable_compl_inf_tol::Float64 = acc_factor * compl_inf_tol

    output_file_name_solver = "ncl_$(ncl.nlp.meta.name)_subproblem.log"

    if print_level_NCL ≥ 1
        @printf(io, "=== %s ===\n", ncl.nlp.meta.name)

        if print_level_NCL ≥ 2
            @printf(io, "Optimization parameters")
                @printf(io, "\n    Global tolerance                 ω_end = %7.2e for gradient lagrangian norm", tol)
                @printf(io, "\n    Global infeasibility             η_end = %7.2e for residuals norm and constraint violation", constr_viol_tol)
                @printf(io, "\n    Global complementarity tolerance ϵ_end = %7.2e for multipliers and constraints", compl_inf_tol)
                @printf(io, "\n    Maximum penalty parameter        ρ_max = %7.2e ", max_penal)
                @printf(io, "\n    Minimum infeasibility accepted   η_min = %7.2e \n", min_infeas)
        end
    end

    # if (nlp.meta.ncon == 0) | ((nlp.meta.nnln == 0)
    #     #** I. Solution with solver
    #     if print_level_NCL ≥ 1
    #         @printf(io, "\n============= Solution of %s with IPOPT (0 residual to add) =============\n", nlp.meta.name)
    #     end
    #
    #     resol = ipopt(nlp;
    #                   print_level = print_level_solver,
    #                   output_file = output_file_name_solver,
    #                   max_iter = max_iter_solver,
    #                   tol = tol,
    #                   constr_viol_tol = constr_viol_tol,
    #                   compl_inf_tol = compl_inf_tol)
    #
    #     resol.solver_specific[:multipliers_con] .= - resol.solver_specific[:multipliers_con] # just to be consistent with our convention
    #
    #     if print_level_NCL ≥ 1
    #         if print_level_NCL ≥ 2
    #             if nlp.meta.ncon != 0
    #                 @printf(io, "    Initial lagrangian gradient : ‖∇Lag(x, λ)‖ = %7.2e (tolerance = %7.1e)\n", norm(grad(nlp, nlp.meta.x0) - jtprod(nlp, nlp.meta.x0, nlp.meta.y0), Inf), tol)
    #             else
    #                 @printf(io, "    Initial lagrangian gradient : ‖∇Lag(x, λ)‖ = %7.2e (tolerance = %7.1e)\n", norm(grad(nlp, nlp.meta.x0), Inf), tol)
    #             end
    #         end
    #
    #         @printf(io, "    Solver conclusion: %s\n", string(resol.solver_specific[:internal_msg]))
    #
    #         if print_level_NCL ≥ 2
    #             @printf(io, "        Number of iterations: %d\n", resol.iter)
    #             println(io, "        Final solution: x = ", resol.solution)
    #             @printf(io, "        Final lagrangian gradient: ‖∇Lag(x, λ)‖ = %7.2e (tolerance = %7.1e)\n", resol.dual_feas, tol)
    #         end
    #     end
    #
    #     return resol
    #
    # else

    #** II. Solution with NCL
    #** II.A. Names and variables
    #** II.A.0 Constants & scale parameters
    mu_init = warm_start_init_point ? 1.0e-3 : 0.1

    τ = 10.0 # scale (used to update the ρ_k step)
    α = 0.1 # Constant (α needs to be < 1)
    β = 0.2 # Constant

    #** II.A.1 Parameters
    ncl.ρ = 100.0 # step
    ρ_max = max_penal # biggest penalization authorized

    ω_end = tol #global tolerance, in argument
    ω_k = ω_end # sub problem initial tolerance

    #! change eta_end
    η_end = constr_viol_tol #global infeasibility in argument
    η_k = 1e-2 # sub problem infeasibility
    η_min = min_infeas # smallest infeasibility authorized

    ϵ_end = compl_inf_tol #global tolerance for complementarity conditions

    acc_count = 0

    #** II.A.2 Initial variables
    x_k = zeros(ncl.nx)
    r_k = zeros(ncl.nr)
    norm_r_k_inf = norm(r_k, Inf) #Pre-computation

    λ_k = zeros(ncl.meta.ncon)
    z_k_U = zeros(length(ncl.meta.uvar))
    z_k_L = zeros(length(ncl.meta.lvar))

    #** II.A.3 Initial print
    if print_level_NCL ≥ 1
            # print(ncl, print_level = print_level_NCL)  #, output_file_print = output_file_print_NCL, output_file = file)

        if print_level_NCL ≥ 2
            @printf(io, "============= NCL Begin =============\n")
            @printf(io, "%5s  %4s  %6s  %7s  %7s  %9s  %7s  %7s  %7s",
                        "Iter", "‖rₖ‖∞", "ηₖ", "ρ", "μ init", "NCL obj", "‖y‖", "‖λₖ‖", "‖xₖ‖")
        end
    end

    print_level_solver > 0 && @printf(io, "\n")

    #** II.B. Optimization loop and return
    k = 0
    converged = false

    while (k ≤ max_iter_NCL) & !converged
        #** II.B.0 Iteration counter and mu_init
        k += 1

        if (k == 2) & warm_start_init_point
            mu_init = 1e-4
        elseif (k == 4) & warm_start_init_point
            mu_init = 1e-5
        elseif (k == 6) & warm_start_init_point
            mu_init = 1e-6
        elseif (k == 8) & warm_start_init_point
            mu_init = 1e-7
        elseif (k == 10) & warm_start_init_point
            mu_init = 1e-8
        end

        #** II.B.1 Get subproblem's solution
        #** II.B.1.1 Solver
        solve_k = ipopt(ncl;
                        tol = ω_k,
                        constr_viol_tol = η_k,
                        compl_inf_tol = ϵ_end,
                        acceptable_tol = acceptable_tol,
                        acceptable_constr_viol_tol = acceptable_constr_viol_tol,
                        acceptable_compl_inf_tol = acceptable_compl_inf_tol,
                        print_level = print_level_solver,
                        output_file=output_file_name_solver,
                        ignore_time = true,
                        warm_start_init_point = warm_start_init_point ? "yes" : "no",
                        mu_init = mu_init,
                        dual_inf_tol=1e-6,
                        max_iter = 1000,
                        sb = "yes")

        # Get variables
        X_k = solve_k.solution #pre-access
        x_k = X_k[1:ncl.nx]
        r_k = X_k[ncl.nx+1 : ncl.nx+ncl.nr]
        norm_r_k_inf = norm(r_k, Inf) # update

        # Get multipliers
        #! Warning, ipopt doesn't use our convention in KKT_check for constraint multipliers, so we took the opposite. For bound multiplier it seems to work though.
        λ_k = - solve_k.solver_specific[:multipliers_con]
        z_k_U = solve_k.solver_specific[:multipliers_U]
        z_k_L = solve_k.solver_specific[:multipliers_L]

        #** II.B.1.2 Output print
        if print_level_NCL ≥ 2
            @printf(io, "%4d  %7.1e  %7.1e  %7.1e  %7.1e  %9.2e  %7.1e  %7.1e  %7.1e",
                        k, norm_r_k_inf, η_k, ncl.ρ, mu_init, obj(ncl, vcat(x_k, r_k)), norm(ncl.y, Inf), norm(λ_k, Inf), norm(x_k))
        end

        print_level_solver > 0 && @printf(io, "\n")

        #** II.B.2 Treatment & update
        if (norm_r_k_inf ≤ max(η_k, η_end)) | (k == max_iter_NCL) # The residual has decreased enough
            #** II.B.2.1 Update
            ncl.y = ncl.y + ncl.ρ * r_k # Updating multiplier
            η_k = max(η_k/τ, η_min) # η_k / (1 + ncl.ρ ^ β) # (heuristic)

            if η_k == η_min
                @warn "Minimum constraint violation η_min = " * string(η_min) * " reached at iteration k = " * string(k)
            end

            #** II.B.2.2 Solution found ?
            if (norm_r_k_inf ≤ η_end) | (k == max_iter_NCL) # check if r_k is small enough, or if we've reached the end
                #* Residual & KKT tests
                if norm_r_k_inf > η_end
                    converged = false
                else
                    if print_level_NCL ≥ 2
                        @printf(io, "\n‖rₖ‖∞ = %7.1e ≤ η_end = %7.1e", norm_r_k_inf, η_end)
                    end

                    if KKT_checking
                        D_solved = KKT_check(ncl.nlp, x_k, λ_k, z_k_U[1:ncl.nx], z_k_L[1:ncl.nx], io, tol=ω_end, constr_viol_tol=η_end, compl_inf_tol=ϵ_end, print_level=print_level_NCL)
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
                dual_feas::Float64 = (ncl.meta.ncon != 0) ? norm(grad(ncl.nlp, x_k) - jtprod(ncl.nlp, x_k, λ_k) - (z_k_L - z_k_U)[1:ncl.nx], Inf) : norm(grad(ncl.nlp, x_k) - (z_k_L - z_k_U)[1:ncl.nx], Inf)
                primal_feas = minimum(vcat(cons(ncl.nlp, x_k) - ncl.nlp.meta.lcon, ncl.nlp.meta.ucon - cons(ncl.nlp, x_k)))

                #* Print results
                if print_level_NCL ≥ 1
                    if converged
                        write(io, "EXIT: optimal solution found\n")
                    elseif acc_count ≥ 3
                        write(io, "EXIT: solved to acceptable level\n")
                    elseif k == max_iter_NCL
                        write(io, "EXIT: reached max_iter_NCL\n")
                    end

                    @printf(io, "============= NCL End =============\n")
                end

                #** II.B.2.3 Return if end of the algorithm
                if converged | (k == max_iter_NCL)
                    return GenericExecutionStats(status, ncl,
                                                solution = x_k,
                                                iter = k,
                                                primal_feas = primal_feas,
                                                dual_feas = dual_feas,
                                                objective = obj(ncl.nlp, x_k),
                                                elapsed_time = 0,
                                                #! doesn't work... counters = nlp.counters,
                                                solver_specific = Dict(:multipliers_con => λ_k,
                                                                       :multipliers_L => z_k_L[1:ncl.nx],
                                                                       :multipliers_U => z_k_U[1:ncl.nx],
                                                                       :internal_msg => converged ? Symbol("Solve_Succeeded") : Symbol("Solve_Failed"),
                                                                       :residuals => r_k
                                                                      )
                                                )
                #else
                #    ncl.ρ = ncl.ρ * τ #! A voir !
                end
            end

        else
            #** II.B.3 Increase penalization
            ncl.ρ = τ * ncl.ρ # increase the step
            #η_k = η_end / (1 + ncl.ρ ^ α) # Change infeasibility (heuristic)
            if ncl.ρ == ρ_max
                @warn "Maximum penalty ρ = " * string(ρ_max) * " reached at iteration k = " * string(k)
            end
        end
        # ? Chez Nocedal & Wright, p.521, on a : ω_k = 1/ncl.ρ, ncl.ρ = 100ρ_k, η_k = 1/ncl.ρ^0.1
    end
end
