import NLPModels: increment!
using NLPModels

mutable struct NCLModel <: AbstractNLPModel
	meta :: NLPModelMeta
	counters :: Counters

	# Parameters for the objective function
	y :: Float64
	ρ :: Float64

	# Sizes for the original problem
	ivar_x :: Array{Int64, 1}
	ivar_r_e :: Array{Int64, 1}
	ivar_r_i :: Array{Int64, 1}
end

function NCLModel(nlp :: NLPModelMeta, mult, penal) :: NCLModel
	meta = NLPModelMeta(nvar = nlp.meta.nvar + nlp.meta.ncon - nlp.meta.lin, 
						lvar = nlp.meta.lvar,
						uvar = nlp.meta.uvar,
						ncon = nlp.meta.ncon, 
						nnzh = nlp.meta.nnzh, 
						nnzj = nlp.meta.nnzj, 
						x0 = nlp.meta.x0, 
						lcon = nlp.meta.lcon, 
						ucon = nlp.meta.ucon, 
						name = nlp.meta.name)
	y = mult
	ρ = penal
	ivar_x = [i for i in 1:nlp.meta.nvar]
	
	
	nvar_x = nlp.meta.nvar
	nvar_r = nlp.meta.ncon - nlp.meta.lin # linear constraints are not considered here in the NCL method. 
	# TODO: another implentation, considering linear constraints as well
	return NCLModel(meta, Counters(), y, ρ)
end


function NLPModels.obj(ncl :: NCLModel, z :: AbstractVector)
	increment!(ncl, :neval_obj)
	return obj(nlp, z[1:NCLModel.nvar_x]) + # ?????? Comment récupérer la fonction objecti du NLP source ?!
		   NCLModel.y' * z[NCLModel.nvar_x + 1 : end] +
		   0.5 * NCLModel.ρ * (norm(z[NCLModel.nvar_x + 1 : end], 2))
end

















function NLPModels.grad!(nlp :: NCLModel, x :: AbstractVector, gx :: AbstractVector)
	increment!(nlp, :neval_grad)
	gx .= [2 * (x[1] - 1); 0.0]
	return gx
end

function NLPModels.hess(nlp :: NCLModel, x :: AbstractVector; obj_weight=1.0, y=Float64[])
	increment!(nlp, :neval_hess)
	w = length(y) > 0 ? y[1] : 0.0
	return [2.0 * obj_weight - 20 * w   0.0; 0.0 0.0]
end
"""
function NLPModels.hess_coord(nlp :: NCLModel, x :: AbstractVector; obj_weight=1.0, y=Float64[])
	increment!(nlp, :neval_hess)
	w = length(y) > 0 ? y[1] : 0.0
	return ([1], [1], [2.0 * obj_weight - 20 * w])
end

function NLPModels.hprod!(nlp :: NCLModel, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight=1.0, y=Float64[])
	increment!(nlp, :neval_hprod)
	w = length(y) > 0 ? y[1] : 0.0
	Hv .= [(2.0 * obj_weight - 20 * w) * v[1]; 0.0]
	return Hv
end
"""
function NLPModels.cons!(nlp :: NCLModel, x :: AbstractVector, cx :: AbstractVector)
	increment!(nlp, :neval_cons)
	cx[1] = 10 * (x[2] - x[1]^2)
	return cx
end

function NLPModels.jac(nlp :: NCLModel, x :: AbstractVector)
	increment!(nlp, :neval_jac)
	return [-20 * x[1]  10.0]
end

function NLPModels.jac_coord(nlp :: NCLModel, x :: AbstractVector)
	increment!(nlp, :neval_jac)
	return ([1, 1], [1, 2], [-20 * x[1], 10.0])
end

function NLPModels.jprod!(nlp :: NCLModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
	increment!(nlp, :neval_jprod)
	Jv .= [-20 * x[1] * v[1] + 10 * v[2]]
	return Jv
end

function NLPModels.jtprod!(nlp :: NCLModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
	increment!(nlp, :neval_jtprod)
	Jtv .= [-20 * x[1]; 10] * v[1]
	return Jtv
end

ncl = ncl()
x = ncl.meta.x0
(obj(ncl, x), hess(ncl, x), grad(ncl, x))
ncl.meta.nvar