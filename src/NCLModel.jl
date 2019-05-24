import NLPModels: increment!
using NLPModels

mutable struct NCLModel <: AbstractNLPModel
	meta :: NLPModelMeta
	counters :: Counters
end

function NCLModel(nlp :: NLPModelMeta) :: NCLModel
	meta = NLPModelMeta(nvar = nlp.meta.nvar, 
						ncon = nlp.meta.ncon, 
						nnzh = nlp.meta.nnzh, 
						nnzj = nlp.meta.nnzj, 
						x0 = nlp.meta.x0, 
						lcon = nlp.meta.lcon, 
						ucon = nlp.meta.ucon, 
						name = nlp.meta.name)

	return NCLModel(meta, Counters())
end


function NLPModels.obj(nlp :: NCLModel, x :: AbstractVector)
	increment!(nlp, :neval_obj)
	return (1 - x[1])^2
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