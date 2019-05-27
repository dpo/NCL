import NLPModels: increment!
using NLPModels
using LinearAlgebra

# TODO : Faire les tests unitaires

"""
Subtype of AbstractNLPModel, adapted to the NCL method. 
Keeps some informations from the original AbstractNLPModel, 
and creates a new problem, modifying 
	the objective function (sort of augmented lagrangian, without considering linear constraints) 
	the constraints (with residues)
"""
mutable struct NCLModel <: AbstractNLPModel
	# Information about the original problem
		nlp::AbstractNLPModel # The original problem
		nvar_x::Int64 # Number of variable of the nlp problem
		nvar_r::Int64 # Number of residues for the nlp problem (in fact nvar_r = size(jres))
		minimize::Bool # true if the aim of the problem is to minimize, false otherwise
		standard::Bool # ? Inutile ???

	# Constant parameters
		meta::AbstractNLPModelMeta # Informations for this problem
		counters::Counters # Counters of calls to functions like obj, grad, cons, of the problem
		nvar::Int64 # Number of variable of this problem (nvar_x + nvar_r)
		jres::Array{Int64, 1} # jres contains the indices of constraints with residues (feasible, not free and not linear)
		
	# Parameters for the objective function
		y::AbstractVector
		ρ::Real
end

function NCLModel(nlp::AbstractNLPModel, mult::AbstractVector, penal::Real, nlp_standard::Bool)::NCLModel
	# Information about the original problem
		nvar_x = nlp.meta.nvar
		nvar_r = nlp.meta.ncon - size(nlp.meta.lin, 1) # linear constraints are not considered here in the NCL method. 
		minimize = nlp.meta.minimize
		standard = nlp_standard
		
	# Constant parameters
		nvar = nvar_x + nvar_r
		meta = NLPModelMeta(nvar ;
							lvar = vcat(nlp.meta.lvar, -Inf * ones(nvar_r)), # No bounds upon residues
							uvar = vcat(nlp.meta.uvar, -Inf * ones(nvar_r)),
							ncon = nlp.meta.ncon, 
							nnzh = nlp.meta.nnzh, 
							nnzj = nlp.meta.nnzj, 
							x0   = vcat(nlp.meta.x0, zeros(Real, nvar_r)),
							lcon = nlp.meta.lcon, 
							ucon = nlp.meta.ucon, 
							name = nlp.meta.name
			)

		jres = []
		for i in 1:nlp.meta.ncon
			if !(i in nlp.meta.jinf) & !(i in nlp.meta.jfree) & !(i in nlp.meta.lin)
				push!(jres, i) 
			end
		end
	
	# Parameters
		y = mult
		ρ = penal

	# NCLModel created:
		return NCLModel(nlp, 
						nvar_x, 
						nvar_r, 
						minimize,
						standard,				
						meta, 
						Counters(), 
						nvar, 
						jres,
						y, 
						ρ)
	# TODO: another implementation, considering linear constraints in the residues, as well. Faire avec un autre sous type
end


function NLPModels.obj(nlc::NCLModel, z::AbstractVector)::Real
	increment!(nlc, :neval_obj)
	if nlc.minimize
		return obj(nlc.nlp, z[1:nlc.nvar_x]) +
		       nlc.y' * z[nlc.nvar_x + 1 : end] +
			   0.5 * nlc.ρ * (norm(z[nlc.nvar_x + 1 : end], 2))
	
	else # argmax f(x) = argmin -f(x)
		nlc.minimize = true 
		return - obj(nlc.nlp, z[1:nlc.nvar_x]) +
			   nlc.y' * z[nlc.nvar_x + 1 : end] +
			   0.5 * nlc.ρ * (norm(z[nlc.nvar_x + 1 : end], 2))
	end
end


function NLPModels.grad(nlc::NCLModel, z::AbstractVector) ::Array{Real, 1}
	increment!(nlc, :neval_grad)
	gx = vcat(grad(nlc.nlp, z[1:nlc.nvar_x]), nlc.ρ * z[nlc.nvar_x+1:end] + nlc.y)
	return gx
end

function NLPModels.grad!(nlc::NCLModel, z::AbstractVector, gx::AbstractVector) ::Array{Real, 1}
	increment!(nlc, :neval_grad)
	gx .= vcat(obj(nlc.nlp, z[1:nlc.nvar_x]), nlc.ρ * z[nlc.nvar_x+1:end] + nlc.y)
	return gx
end

# TODO (simple): sparse, pas matrice complète
function NLPModels.hess(nlc::NCLModel, z::AbstractVector ; obj_weight=1.0, y=zeros) ::Array{Real, 2}
	increment!(nlc, :neval_hess)
	H = zeros(nlc.nvar, nlc.nvar)
	# Original information
		H[1:nlc.nvar_x, 1:nlc.nvar_x] = hess(nlc.nlp, z[1:nlc.nvar_x], obj_weight=obj_weight, y=y) # Original hessian
	
	# New information (due to residues)
		H[nlc.nvar_x+1:end, nlc.nvar_x+1:end] = H[nlc.nvar_x+1:end, nlc.nvar_x+1:end] + nlc.ρ * I # Added by residues (constant because of quadratic penalization) 
	
	# TODO: check there is no problem with the I::Uniformscaling operator (size...)
	return H
end

function NLPModels.hess_coord(nlc::NCLModel, z::AbstractVector ; obj_weight=1.0, y=zeros)
	increment!(nlc, :neval_hess)
	# Original information
		rows, cols, vals = hess_coord(nlc.nlp, z[1:nlc.nvar_x], obj_weight=obj_weight, y=y)
	
	# New information (due to residues)
		rows = vcat(rows, nlc.nvar_x+1:nlc.nvar)
		cols = vcat(cols, nlc.nvar_x+1:nlc.nvar)
		vals = vcat(vals, fill!(Vector{Real}(undef, nlc.nvar_r), nlc.ρ)) # concatenate with a vector full of nlc.ρ
	return (rows, cols, vals)
end


function NLPModels.hprod(nlc::NCLModel, x::AbstractVector, v::AbstractVector ; obj_weight=1.0, y=zeros) ::Array{Real, 1}
	increment!(nlc, :neval_hprod)
	# Test feasability
		if size(v) != nlc.nvar
			println("ERROR: Wrong sizes of argument v passed to jprod\n
					Empty vector returned")
			return []
		end


	# Original information
		nlp_Hv = hprod(nlc.nlp, x, v[1:nlc.nvar_x], obj_weight=obj_weight, y=y)
	
	# New information (due to residues)
		Hv = vcat(nlp_Hv, nlc.ρ * v[nlc.nvar_x+1:end])
	
	return Hv
end



function NLPModels.cons(nlc::NCLModel, z::AbstractVector) ::Array{Real, 1}
	increment!(nlc, :neval_cons)
	# Original information
		cx = cons(nlc.nlp, z[1:nlc.nvar_x]) # pre computation

	# New information (due to residues)
		for i in nlc.jres
			cx[i] += z[nlc.nvar_x + i] # residue for the i-th constraint (feasible, not free and not linear (not considered in this model))
		end
	
	return cx
end


function NLPModels.cons!(nlc::NCLModel, z::AbstractVector, cx::AbstractVector) ::Array{Real, 1}
	increment!(nlc, :neval_cons)
	# Original information
		cx = cons(nlc.nlp, z[1:nlc.nvar_x]) # pre computation

	# New information (due to residues)
		for i in nlc.jres
			cx[i] += z[nlc.nvar_x + i] # residue for the i-th constraint (feasible, not free and not linear (not considered in this model))
		end

	return cx
end

# TODO (simple): return sparse, pas matrice complète
function NLPModels.jac(nlc::NCLModel, z::AbstractVector) ::Array{Real, 2}
	increment!(nlc, :neval_jac)
	# Original information
		J = jac(nlc.nlp, z[1:nlc.nvar_x])

	# New information (due to residues)
		J = vcat(J, I) # residues part
		J = J[:, nlc.jres] # but some constraint don't have a residue, so we remove some
	return J
end

function NLPModels.jac_coord(nlc::NCLModel, z::AbstractVector)
	increment!(nlc, :neval_jac)
	# Original information
		rows, cols, vals = jac_coord(nlc.nlp, z[1:nlc.nvar_x])

	# New information (due to residues)
		rows = vcat(rows, 1:nlc.meta.ncon)
		cols = vcat(cols, nlc.nvar_x+1 : nlc.nvar_r)
		vals = vcat(vals, ones(Real, nlc.nvar_r))
	
	return rows, cols, vals
end

function NLPModels.jprod!(nlc::NCLModel, z::AbstractVector, v::AbstractVector, Jv::AbstractVector) ::Array{Real, 1}
	increment!(nlc, :neval_jprod)
	# Test feasability
		if size(v) != nlc.nvar
			println("ERROR: Wrong sizes of argument v passed to jprod\n
					Empty vector returned")
			return []
		end

	# Original information
		jprod!(nlc.nlp, z[1:nlc.nvar_x], v[1:nlc.nvar_x], Jv[1:nlc.nvar_x])
	
	# New information (due to residues)
		Jv = vcat(Jv, v[nlc.nvar_x+1:end])

	return Jv
end

function NLPModels.jtprod!(nlc::NCLModel, z::AbstractVector, v::AbstractVector, Jtv::AbstractVector) ::Array{Real, 1}
	increment!(nlc, :neval_jtprod)
	# Test feasability
		if size(v) != nlc.ncon
			println("ERROR: Wrong sizes of argument v passed to jprod\n
					Empty vector returned")
			return []
		end

	# New information (due to residues)
		Resv = []
		for i in 1:nlc.ncon #? A tester !
			if i in nlc.jres
				push!(Resv, v[i]) 
			else
				push!(Resv, 0)
			end
		end

	# Original information
		Jv = jtprod(nlc.nlp, z[1:nlc.nvar_x], v) + Resv

	return Jv
end





f(x) = x[1]
x0 = [1]
lvar = [-1]
uvar = [12]
lcon = [0,0]
ucon = [Inf,Inf]
c(x) = [2*x[1], 3*x[1]]
nlp = ADNLPModel(f, x0; lvar=lvar, uvar=uvar, c=c, lcon=lcon, ucon=ucon)


nlc = NCLModel(nlp, [0.,0.], 1., false)::NCLModel

x = nlc.meta.x0
obj(nlc, x)
hess(nlc, [0.,0.,0.], obj_weight = 1., y = [0.,0.])
grad(nlc, x)
