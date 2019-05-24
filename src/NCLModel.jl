import NLPModels: increment!
using NLPModels
using LinearAlgebra

mutable struct NLCModel <: AbstractNLPModel
	meta::NLPModelMeta
	counters::Counters

	# Parameters for the objective function
	y::AbstractVector
	ρ::Number

	# Informations of the original problem
	nvar_x::Int64
	nvar_r::Int64
	f_obj::Function
	∇f_obj::Function
end

function NLCModel(nlp::ADNLPModel, mult::AbstractVector, penal::Number)::NLCModel
	@show nlp.meta.nvar, nlp.meta.ncon, size(nlp.meta.lin, 1)

	y = mult
	ρ = penal
	f_obj(x), ∇f_obj(x) = objgrad(nlp, x)
	nvar_x = nlp.meta.nvar
	nvar_r = nlp.meta.ncon - size(nlp.meta.lin, 1) # linear constraints are not considered here in the NCL method. 
	# TODO: another implentation, considering linear constraints as well

	meta = NLPModelMeta(nlp.meta.nvar + nlp.meta.ncon - size(nlp.meta.lin, 1), ;
						lvar = vcat(nlp.meta.lvar, -Inf * ones(nvar_r)),
						uvar = vcat(nlp.meta.uvar, -Inf * ones(nvar_r)),
						ncon = nlp.meta.ncon, 
						nnzh = nlp.meta.nnzh, 
						nnzj = nlp.meta.nnzj, 
						x0 = vcat(nlp.meta.x0, zeros(Float64, nvar_r)),
						lcon = nlp.meta.lcon, 
						ucon = nlp.meta.ucon, 
						name = nlp.meta.name)

	return NLCModel(meta, Counters(), y, ρ, nvar_x, nvar_r, f_obj, ∇f_obj)
end


function NLPModels.obj(nlc::NLCModel, z::AbstractVector)
	increment!(ncl, :neval_obj)

	return nlc.f_obj(z[1:nlc.nvar_x]) + # ?????? Comment récupérer la fonction objectif du NLP source ?!
		   nlc.y' * z[nlc.nvar_x + 1 : end] +
		   0.5 * nlc.ρ * (norm(z[nlc.nvar_x + 1 : end], 2))
end


function NLPModels.grad!(nlc::NLCModel, z::AbstractVector, gx::AbstractVector)
	increment!(nlc, :neval_grad)
	gx .= vcat(∇f_obj(z[1:nlc.nvar_x]), nlc.ρ * z[nlc.nvar_x+1:end] + nlc.y)
	return gx
end


function NLPModels.hess(nlc::NLCModel, x::AbstractVector)
	increment!(nlc, :neval_hess)
	
	return 
end
"""
function NLPModels.hess_coord(nlc::NLCModel, x::AbstractVector; obj_weight=1.0, y=Float64[])
	increment!(nlc, :neval_hess)
	w = length(y) > 0 ? y[1] : 0.0
	return ([1], [1], [2.0 * obj_weight - 20 * w])
end

function NLPModels.hprod!(nlc::NLCModel, x::AbstractVector, v::AbstractVector, Hv::AbstractVector; obj_weight=1.0, y=Float64[])
	increment!(nlc, :neval_hprod)
	w = length(y) > 0 ? y[1] : 0.0
	Hv .= [(2.0 * obj_weight - 20 * w) * v[1]; 0.0]
	return Hv
end
"""
function NLPModels.cons!(nlc::NLCModel, x::AbstractVector, cx::AbstractVector)
	increment!(nlc, :neval_cons)
	cx[1] = 10 * (x[2] - x[1]^2)
	return cx
end

function NLPModels.jac(nlc::NLCModel, x::AbstractVector)
	increment!(nlc, :neval_jac)
	return [-20 * x[1]  10.0]
end

function NLPModels.jac_coord(nlc::NLCModel, x::AbstractVector)
	increment!(nlc, :neval_jac)
	return ([1, 1], [1, 2], [-20 * x[1], 10.0])
end

function NLPModels.jprod!(nlc::NLCModel, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
	increment!(nlc, :neval_jprod)
	Jv .= [-20 * x[1] * v[1] + 10 * v[2]]
	return Jv
end

function NLPModels.jtprod!(nlc::NLCModel, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector)
	increment!(nlc, :neval_jtprod)
	Jtv .= [-20 * x[1]; 10] * v[1]
	return Jtv
end





f(x) = x[1]
x0 = [1]
lvar = [-1]
uvar = [12]
lcon = [0,0]
ucon = [Inf,Inf]
c(x) = [2*x[1], 3*x[1]]
nlp = ADNLPModel(f, x0; lvar=lvar, uvar=uvar, c=c, lcon=lcon, ucon=ucon)




ncl = NLCModel(nlp, [0.0,0.0], 0.)
x = ncl.meta.x0
(obj(ncl, x), hess(ncl, x), grad(ncl, x))
ncl.meta.nvar