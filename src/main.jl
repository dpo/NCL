using NLPModels
include("ncl.jl")
include("NLCModel.jl")

"""
Main function for the NCL method. 
    Takes an AbstractNLPModel as initial problem, 
    Converts it to a NLCModel,
    Calls ncl.jl on it,
    Returns (x (solution), y (lagrangian multipliers for constraints), z (lagrangian multpliers for bound constraints))
"""
function main(nlp::AbstractNLPModel) ::Tuple{Vector{<:Real}, Vector{<:Real}, Vector{<:Real}} 
end