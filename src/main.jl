using NLPModels
using Ipopt
using NLPModelsIpopt

include("ncl.jl")
include("NLCModel.jl")

"""
Main function for the NCL method. 
    Takes an AbstractNLPModel as initial problem, 
    Converts it to a NLCModel,
    Calls ncl.jl on it,
    Returns (x (solution), y (lagrangian multipliers for constraints), z (lagrangian multpliers for bound constraints))
"""
function NCLMain(nlp::AbstractNLPModel; tol = 0.01::Real, max_iter = 200::Int64, use_ipopt = true::Bool, printing_iterations = false::Bool, printing_iterations_solver = false::Bool, printing_check = false::Bool) ::Tuple{GenericExecutionStats, Bool}
    if nlp.meta.ncon == 0
        if use_ipopt
            return ipopt(nlp, tol = tol, max_iter = max_iter)
        else
            return _knitro(nlp, tol = tol, max_iter = max_iter)
        end
    else
        nlc = NLCModel(nlp)
        resol = NCLSolve(nlc, max_iter, use_ipopt, tol, printing_iterations::Bool, printing_iterations_solver::Bool, printing_check::Bool)
        
        x = resol.solution[1:nlc.nvar_x] # Warning, NCLSolve returns [x, r], it is the solution of nlc, but oncly x is the the solution of nlp
        λ = resol.solver_specific[:multipliers_con]
        z_U = resol.solver_specific[:multipliers_U][1:nlc.nvar_x] # same for bounds multipliers
        z_L = resol.solver_specific[:multipliers_L][1:nlc.nvar_x]

        optimal = NLPModel_solved(nlp, x, λ, z_U, z_L, tol, printing_check)
        return (resol, optimal)
    end
end