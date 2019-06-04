using NLPModels
using Ipopt
using NLPModelsIpopt

include("ncl.jl")
include("NLCModel.jl")



printing = true
"""
Main function for the NCL method. 
    Takes an AbstractNLPModel as initial problem, 
    Converts it to a NLCModel,
    Calls ncl.jl on it,
    Returns (x (solution), y (lagrangian multipliers for constraints), z (lagrangian multpliers for bound constraints))
"""
#TODO kwargs... + surcharge NCLSolve
function NCLMain(nlp::AbstractNLPModel; tol::Real = 0.01, constr_viol_tol = 0.001::Real, compl_inf_tol = 0.001::Real, max_iter = 200::Int64, use_ipopt = true::Bool, printing_iterations = printing::Bool, printing_iterations_solver = false::Bool, printing_check = printing::Bool) ::Tuple{GenericExecutionStats, Bool} 
    if (nlp.meta.ncon == 0) | (nlp.meta.nnln == 0)
        if printing_iterations
            println("Résolution de " * nlp.meta.name * " par IPOPT / KNITRO")
        end

        if use_ipopt
            return (NLPModelsIpopt.ipopt(nlp, tol=tol, constr_viol_tol=constr_viol_tol, compl_inf_tol=compl_inf_tol, max_iter=max_iter, print_level=printing_iterations_solver ? 3 : 0), true)
        else
            return (_knitro(nlp, tol = tol, constr_viol_tol=constr_viol_tol, compl_inf_tol=compl_inf_tol, max_iter = max_iter, print_level = printing_iterations_solver ? 3 : 0), true)
        end

    else
        nlc = NLCModel(nlp, printing = printing_iterations)
        if printing_iterations
            println("\n")
        end

        resol = NCLSolve(nlc, max_iter, use_ipopt, tol, constr_viol_tol, compl_inf_tol, printing_iterations, printing_iterations_solver, printing_check)
        if printing_iterations
            println("\n")
        end
        
        x = resol.solution[1:nlc.nvar_x] # Warning, NCLSolve returns [x, r], it is the solution of nlc, but oncly x is the the solution of nlp
        λ = resol.solver_specific[:multipliers_con]
        z_U = resol.solver_specific[:multipliers_U][1:nlc.nvar_x] # same for bounds multipliers
        z_L = resol.solver_specific[:multipliers_L][1:nlc.nvar_x]

        optimal = !(resol.iter == max_iter)

        return (resol, optimal)
    end
end