import NLPModels: increment!
using NLPModels
using LinearAlgebra

"""
Subtype of AbstractNLPModel, adapted to the NCL method. 
Keeps some inormations from the original ADNLPModel, but 
"""
mutable struct NCLModel_ <: AbstractNLPModel
	meta::NLPModelMeta
	counters::Counters
	nvar::Int64

	# Parameters for the objective function
	y::AbstractVector
	ρ::Number

	# Information about the original problem
	nvar_x::Int64
	nvar_r::Int64
	nlp_f_obj::Function
	∇nlp_f_obj::Function
	nlp_cons::Function
	nlp_hess_lag::Function
	nlp_hess_coord::Function
	nlp_hprod::Function
	minimize::Bool
	standard::Bool
end

function NCLModel_(nlp::ADNLPModel, mult::AbstractVector, penal::Number, nlp_standard::Bool)::NCLModel_
	nvar = nlp.meta.nvar + nlp.meta.ncon - size(nlp.meta.lin, 1)
	
	# Parameters
	y = mult
	ρ = penal

	# Previous model
	
	nlp_cons(x) = cons(nlp, x)
	nlp_f_obj(x) = obj(nlp, x)
	∇nlp_f_obj(x) = grad(nlp, x)
	nlp_hess_lag(x, obj_weight, y) = hess(nlp, x ; obj_weight = obj_weight, y = y)
	nlp_hess_coord(x, obj_weight, y) = hess_coord(nlp, x ; obj_weight = obj_weight, y = y)
	nlp_hprod(nlp, x, v, obj_weight, y) = hprod(nlp, x, v ; obj_weight = obj_weight, y = y)
	nvar_x = nlp.meta.nvar
	nvar_r = nlp.meta.ncon - size(nlp.meta.lin, 1) # linear constraints are not considered here in the NCL method. 
	minimize = nlp.meta.minimize
	standard = nlp_standard
	# TODO: another implementation, considering linear constraints in the residues, as well


	# Meta field
	meta = NLPModelMeta(nlp.meta.nvar + nvar_r ;
						lvar = vcat(nlp.meta.lvar, -Inf * ones(nvar_r)), # No bounds upon residues
						uvar = vcat(nlp.meta.uvar, -Inf * ones(nvar_r)),
						ncon = nlp.meta.ncon, 
						nnzh = nlp.meta.nnzh, 
						nnzj = nlp.meta.nnzj, 
						x0 = vcat(nlp.meta.x0, zeros(Float64, nvar_r)),
						lcon = nlp.meta.lcon, 
						ucon = nlp.meta.ucon, 
						name = nlp.meta.name)

	return NCLModel_(meta, 
					Counters(), 
					nvar, 
					y, 
					ρ, 
					nvar_x, 
					nvar_r, 
					nlp_f_obj, 
					∇nlp_f_obj, 
					nlp_cons, 
					nlp_hess_lag, 
					nlp_hess_coord,
					nlp_hprod,
					minimize,
					standard)
end


function NLPModels.obj(nlc::NCLModel_, z::AbstractVector)
	increment!(ncl, :neval_obj)
	if nlc.minimize
		return nlc.nlp_f_obj(z[1:nlc.nvar_x]) +
		       nlc.y' * z[nlc.nvar_x + 1 : end] +
			   0.5 * nlc.ρ * (norm(z[nlc.nvar_x + 1 : end], 2))
	
	else # argmax f(x) = argmin -f(x)
		nlc.minimize = true 
		return - nlc.nlp_f_obj(z[1:nlc.nvar_x]) +
			   nlc.y' * z[nlc.nvar_x + 1 : end] +
			   0.5 * nlc.ρ * (norm(z[nlc.nvar_x + 1 : end], 2))
	end
end


function NLPModels.grad(nlc::NCLModel_, z::AbstractVector)
	increment!(nlc, :neval_grad)
	gx = vcat(nlc.∇nlp_f_obj(z[1:nlc.nvar_x]), nlc.ρ * z[nlc.nvar_x+1:end] + nlc.y)
	return gx
end

function NLPModels.grad!(nlc::NCLModel_, z::AbstractVector, gx::AbstractVector)
	increment!(nlc, :neval_grad)
	gx .= vcat(nlc.∇nlp_f_obj(z[1:nlc.nvar_x]), nlc.ρ * z[nlc.nvar_x+1:end] + nlc.y)
	return gx
end


function NLPModels.hess(nlc::NCLModel_, z::AbstractVector ; obj_weight=1.0, y=zeros)
	increment!(nlc, :neval_hess)
	H = zeros(nlc.nvar, nlc.nvar) # = nvar_x + nvar_r -
	H[1:nlc.nvar_x, 1:nlc.nvar_x] = nlc.nlp_hess_lag(z[1:nlc.nvar_x], obj_weight, y) # Original hessian
	H[nlc.nvar_x+1:end, nlc.nvar_x+1:end] = H[nlc.nvar_x+1:end, nlc.nvar_x+1:end] + nlc.ρ * I # Added by residues (constant because of quadratic penalization) 
	# TODO: check there is no problem with the I::Uniformscaling operator (size...)
	return H
end

function NLPModels.hess_coord(nlc::NCLModel_, z::AbstractVector ; obj_weight=1.0, y=zeros)
	increment!(nlc, :neval_hess)
	rows, cols, vals = nlc.hess_coord(z[1:ncl.nvar_x], obj_weight, y)
	rows = vcat(rows, ncl.nvar_x+1:ncl.nvar)
	cols = vcat(cols, ncl.nvar_x+1:ncl.nvar)
	vals = vcat(vals, ncl.ρ * ones(ncl.nvar_r))
	return (rows, cols, vals)
end

function NLPModels.hprod(nlc::NCLModel_, x::AbstractVector, v::AbstractVector ; obj_weight=1.0, y=zeros)
	increment!(nlc, :neval_hprod)
	nlp_Hv = 
	Hv = zeros(nlc.nvar, 1)
	Hv[1:ncl.nvar_x] = ncl.nlp_hprod(x, v[1:ncl.nvar_x], obj_weight, y)
	Hv[ncl.nvar_x+1:end] = ncl.ρ * v[ncl.nvar_x+1:end]
	return Hv
end




function NLPModels.cons(nlc::NCLModel_, x::AbstractVector)
	increment!(nlc, :neval_cons)
	cx = 0
	# Tri
	# Standardisation + résidus
	return cx
end


function NLPModels.cons!(nlc::NCLModel_, x::AbstractVector, cx::AbstractVector)
	increment!(nlc, :neval_cons)
	cx[1] = 10 * (x[2] - x[1]^2)
	return cx
end

function NLPModels.jac(nlc::NCLModel_, x::AbstractVector)
	increment!(nlc, :neval_jac)
	return [-20 * x[1]  10.0]
end

function NLPModels.jac_coord(nlc::NCLModel_, x::AbstractVector)
	increment!(nlc, :neval_jac)
	return ([1, 1], [1, 2], [-20 * x[1], 10.0])
end

function NLPModels.jprod!(nlc::NCLModel_, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
	increment!(nlc, :neval_jprod)
	Jv .= [-20 * x[1] * v[1] + 10 * v[2]]
	return Jv
end

function NLPModels.jtprod!(nlc::NCLModel_, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector)
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




ncl = NCLModel_(nlp, [0.,0.], 1., false)::NCLModel_

x = ncl.meta.x0
obj(ncl, x)
hess(ncl, [0.,0.,0.], obj_weight = 1., y = [0.,0.])
grad(ncl, x)
