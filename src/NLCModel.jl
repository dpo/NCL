import NLPModels: increment!
using NLPModels
using LinearAlgebra
using Test

# TODO : Faire les tests unitaires

"""
Subtype of AbstractNLPModel, adapted to the NCL method. 
Keeps some informations from the original AbstractNLPModel, 
and creates a new problem, modifying 
	the objective function (sort of augmented lagrangian, without considering linear constraints) 
	the constraints (with residues)
"""
mutable struct NLCModel <: AbstractNLPModel
	# Information about the original problem
		nlp::AbstractNLPModel # The original problem
		nvar_x::Int64 # Number of variable of the nlp problem
		nvar_r::Int64 # Number of residues for the nlp problem (in fact nvar_r = size(nln,1), if there are no free/infeasible constraints)
		minimize::Bool # true if the aim of the problem is to minimize, false otherwise
		jres::Vector{Int64} # Vector of indices of the non linear constraints (implemented only if nlp.meta.lin is empty)

	# Constant parameters
		meta::AbstractNLPModelMeta # Informations for this problem
		counters::Counters # Counters of calls to functions like obj, grad, cons, of the problem
		nvar::Int64 # Number of variable of this problem (nvar_x + nvar_r)
		
	# Parameters for the objective function
		y::Vector{<:Real}
		ρ::Real
end

function NLCModel(nlp::AbstractNLPModel, mult::Vector{<:Real}, penal::Real)::NLCModel
	# Information about the original problem
		if (nlp.meta.lin == Int[]) & (isa(nlp, ADNLPModel)) & (nlp.meta.name == "Unitary test problem")
			jres = [2, 4] 
			nvar_r = 2 # linear constraints are not considered here in the NCL method. 
		else
			nvar_r = nlp.meta.nnln # linear constraints are not considered here in the NCL method. 
			jres = nlp.meta.nln # copy, useless, but permits to use the unitary test problem computed				
		end

		nvar_x = nlp.meta.nvar
		minimize = nlp.meta.minimize
		
	# Constant parameters
		nvar = nvar_x + nvar_r
		meta = NLPModelMeta(nvar ;
							lvar = vcat(nlp.meta.lvar, -Inf * ones(nvar_r)), # No bounds upon residues
							uvar = vcat(nlp.meta.uvar, Inf * ones(nvar_r)),
							x0   = vcat(nlp.meta.x0, ones(typeof(nlp.meta.x0[1]), nvar_r)),
							name = nlp.meta.name * " (NCL subproblem)",
							nnzj = nlp.meta.nnzj + nvar_r, # we add nonzeros because of residues
							nnzh = nlp.meta.nnzh + nvar_r,
							)

		if nlp.meta.jinf != Int64[]
			error("argument problem passed to NLCModel with constraint " * string(nlp.meta.jinf) * " infeasible")
		end
		if nlp.meta.jfree != Int64[]
			error("argument problem passed to NLCModel with constraint " * string(nlp.meta.jfree) * " free")
		end
	
	# Parameters
		y = mult
		ρ = penal

	# NLCModel created:
		return NLCModel(nlp, 
						nvar_x, 
						nvar_r, 
						minimize,	
						jres,		
						meta, 
						Counters(), 
						nvar,
						y, 
						ρ)
end


function NLPModels.obj(nlc::NLCModel, X::Vector{<:Real})::Real
	increment!(nlc, :neval_obj)
	if nlc.minimize
		if nlc.nvar_r == 0 # little test to avoid []' * []
			return obj(nlc.nlp, X[1:nlc.nvar_x])
		else
			return obj(nlc.nlp, X[1:nlc.nvar_x]) +
				nlc.y' * X[nlc.nvar_x + 1 : end] +
				0.5 * nlc.ρ * (norm(X[nlc.nvar_x + 1 : end], 2) ^ 2)
		end

	else # argmax f(x) = argmin -f(x)
		nlc.minimize = true 
		if nlc.nvar_r == 0 
			return - obj(nlc.nlp, X[1:nlc.nvar_x])
		else
			return - obj(nlc.nlp, X[1:nlc.nvar_x]) +
				nlc.y' * X[nlc.nvar_x + 1 : end] +
				0.5 * nlc.ρ * (norm(X[nlc.nvar_x + 1 : end], 2) ^ 2)
		end
	end
end



function NLPModels.grad(nlc::NLCModel, X::Vector{<:Real}) ::Vector{<:Real}
	increment!(nlc, :neval_grad)
	gx = vcat(grad(nlc.nlp, X[1:nlc.nvar_x]), nlc.ρ * X[nlc.nvar_x+1:end] + nlc.y)
	return gx
end

function NLPModels.grad!(nlc::NLCModel, X::Vector{<:Real}, gx::Vector{<:Real}) ::Vector{<:Real}
	increment!(nlc, :neval_grad)
	if size(gx, 1) != nlc.nvar
		println("ERROR: wrong size of argument gx passed to grad! in NLCModel
				 gx should be of size " * string(nlc.nvar) * " but size " * string(size(gx)) * 
				 "given
				 Empty vector returned")
		return <:Real[]
	end

	gx .= vcat(grad(nlc.nlp, X[1:nlc.nvar_x]), nlc.ρ * X[nlc.nvar_x+1:end] + nlc.y)
	return gx
end



# TODO (simple): sparse du triangle inf, pas matrice complète
function NLPModels.hess(nlc::NLCModel, X::Vector{<:Real} ; obj_weight=1.0, y=zeros) ::Matrix{<:Real}
	increment!(nlc, :neval_hess)
	H = zeros(nlc.nvar, nlc.nvar)
	# Original information
		H[1:nlc.nvar_x, 1:nlc.nvar_x] = hess(nlc.nlp, X[1:nlc.nvar_x], obj_weight=obj_weight, y=y) # Original hessian
	
	# New information (due to residues)
		H[nlc.nvar_x+1:end, nlc.nvar_x+1:end] = H[nlc.nvar_x+1:end, nlc.nvar_x+1:end] + nlc.ρ * I # Added by residues (constant because of quadratic penalization) 
	
	# TODO: check there is no problem with the I::Uniformscaling operator (size...)
	return H
end

function NLPModels.hess_coord(nlc::NLCModel, X::Vector{<:Real} ; obj_weight=1.0, y=zeros) #? type de retour ?
	increment!(nlc, :neval_hess)
	# Original information
		rows, cols, vals = hess_coord(nlc.nlp, X[1:nlc.nvar_x], obj_weight=obj_weight, y=y)
	
	# New information (due to residues)
		rows = vcat(rows, nlc.nvar_x+1:nlc.nvar)
		cols = vcat(cols, nlc.nvar_x+1:nlc.nvar)
		vals = vcat(vals, fill!(Vector{typeof(nlc.ρ)}(undef, nlc.nvar_r), nlc.ρ)) # concatenate with a vector full of nlc.ρ
	return (rows, cols, vals)
end

function NLPModels.hprod(nlc::NLCModel, X::Vector{<:Real}, v::Vector{<:Real} ; obj_weight=1.0, y=zeros) ::Vector{<:Real}
	increment!(nlc, :neval_hprod)
	# Test feasability
		if size(v, 1) != nlc.nvar
			error("wrong size of argument v passed to hprod in NLCModel
				   gx should be of size " * string(nlc.nvar) * " but size " * string(size(v, 1)) * "given")
		end

	# Original information
		nlp_Hv = hprod(nlc.nlp, X[1:nlc.nvar_x], v[1:nlc.nvar_x], obj_weight=obj_weight, y=y)
	
	# New information (due to residues)
		Hv = vcat(nlp_Hv, nlc.ρ * v[nlc.nvar_x+1:end])
	
	return Hv
end



function NLPModels.cons(nlc::NLCModel, X::Vector{<:Real}) ::Vector{<:Real}
	increment!(nlc, :neval_cons)
	# Original information
		cx = cons(nlc.nlp, X[1:nlc.nvar_x]) # pre computation

	# New information (due to residues)
		cx[nlc.jres] += X[nlc.nvar_x+1:end] # residue for the i-th constraint (feasible, not free and not linear (not considered in this model))
	
	return cx
end

function NLPModels.cons!(nlc::NLCModel, X::Vector{<:Real}, cx::Vector{<:Real}) ::Vector{<:Real}
	increment!(nlc, :neval_cons)
	# Original information
		cx .= cons!(nlc.nlp, X[1:nlc.nvar_x], cx) # pre computation

	# New information (due to residues)
		cx[nlc.jres] .+= X[nlc.nvar_x+1:end] # residue for the i-th constraint (feasible, not free and not linear (not considered in this model))

	return cx
end

# TODO (simple): return sparse, pas matrice complète
function NLPModels.jac(nlc::NLCModel, X::Vector{<:Real}) ::Matrix{<:Real}
	increment!(nlc, :neval_jac)
	# Original information
		J = jac(nlc.nlp, X[1:nlc.nvar_x])
		
	# New information (due to residues)
		J = hcat(J, I) # residues part
		J = J[1:end, vcat(1:nlc.nvar_x, nlc.jres .+ nlc.nvar_x)] # but some constraint don't have a residue, so we remove some
		
	return J
end

function NLPModels.jac_coord(nlc::NLCModel, X::Vector{<:Real}) #? type de retour ?
	increment!(nlc, :neval_jac)
	# Original information
		rows, cols, vals = jac_coord(nlc.nlp, X[1:nlc.nvar_x])

	# New information (due to residues)
		rows = vcat(rows, nlc.jres)
		cols = vcat(cols, nlc.nvar_x+1 : nlc.nvar)
		vals = vcat(vals, ones(typeof(vals[1]), nlc.nvar_r, 1))
	
	return rows, cols, vals
end

function NLPModels.jprod(nlc::NLCModel, X::Vector{<:Real}, v::Vector{<:Real}) ::Vector{<:Real}
	increment!(nlc, :neval_jprod)
	# Test feasability
		if size(v,1) != nlc.nvar
			error("wrong sizes of argument v passed to jprod(nlc::NLCModel, X::Vector{<:Real}, v::Vector{<:Real}, Jv::Vector{<:Real}) ::Vector{<:Real}")
		end
		
	# Original information
		Jv = jprod(nlc.nlp, X[1:nlc.nvar_x], v[1:nlc.nvar_x])

	# New information (due to residues)
		Resv = zeros(typeof(Jv[1,1]), nlc.nlp.meta.ncon)
		Resv[nlc.jres] = Resv[nlc.jres] + v[nlc.nvar_x+1:end]

	return Jv + Resv
end

function NLPModels.jprod!(nlc::NLCModel, X::Vector{<:Real}, v::Vector{<:Real}, Jv::Vector{<:Real}) ::Vector{<:Real}
	increment!(nlc, :neval_jprod)
	# Test feasability
		if size(v,1) != nlc.nvar
			error("wrong sizes of argument v passed to jprod!(nlc::NLCModel, X::Vector{<:Real}, v::Vector{<:Real}, Jv::Vector{<:Real}) ::Vector{<:Real}")
		end

	# Original information
		Jv .= jprod(nlc.nlp, X[1:nlc.nvar_x], v[1:nlc.nvar_x])
		
	# New information (due to residues)
		Resv = zeros(typeof(Jv[1,1]), nlc.nlp.meta.ncon)
		Resv[nlc.jres] .+= v[nlc.nvar_x+1:end]
		Jv .+= Resv
		
	return Jv
end

function NLPModels.jtprod(nlc::NLCModel, X::Vector{<:Real}, v::Vector{<:Real}) ::Vector{<:Real}
	increment!(nlc, :neval_jtprod)
	# Test feasability
		if size(v,1) != nlc.nlp.meta.ncon
			error("wrong size of argument v passed to jtprod(nlc::NLCModel, X::Vector{<:Real}, v::Vector{<:Real}, Jtv::Vector{<:Real}) ::Vector{<:Real}")
		end

	# New information (due to residues)
		Resv = v[nlc.jres]
	
		@show Resv
		@show jtprod(nlc.nlp, X[1:nlc.nvar_x], v)
	# Original information
		Jv = vcat(jtprod(nlc.nlp, X[1:nlc.nvar_x], v), Resv)
	
	return Jv
end

function NLPModels.jtprod!(nlc::NLCModel, X::Vector{<:Real}, v::Vector{<:Real}, Jtv::Vector{<:Real}) ::Vector{<:Real}
	increment!(nlc, :neval_jtprod)
	# Test feasability
	if size(v,1) != nlc.nlp.meta.ncon
		error("wrong size of argument v passed to jtprod(nlc::NLCModel, X::Vector{<:Real}, v::Vector{<:Real}, Jtv::Vector{<:Real}) ::Vector{<:Real}")
	end

	# New information (due to residues)
		Resv = v[nlc.jres]

	# Original information
		Jtv .= vcat(jtprod(nlc.nlp, X[1:nlc.nvar_x], v), Resv)

	return Jtv
end




function update_y(nlc, y_new)
	nlc.y = y_new
end

function update_ρ(nlc, ρ_new)
	nlc.ρ = ρ_new
end


