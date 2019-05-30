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
function NCLMain(nlp::AbstractNLPModel; tol = 0.01::Real, max_iter = 200::Int64, use_ipopt = true::Bool) ::Tuple{Vector{<:Real}, Vector{<:Real}, Vector{<:Real}} 
    if nlp.meta.ncon == 0
        if use_ipopt
            return ipopt(nlp, tol = tol, max_iter = max_iter)
        else
            return _knitro(nlp, tol = tol, max_iter = max_iter)
        end
    else
        nlc = NLCModel(nlp)
    end
end