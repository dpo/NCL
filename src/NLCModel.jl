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
		nvar_r::Int64 # Number of residues for the nlp problem (in fact nvar_r = size(jres))
		minimize::Bool # true if the aim of the problem is to minimize, false otherwise
		standard::Bool # ? Inutile ???

	# Constant parameters
		meta::AbstractNLPModelMeta # Informations for this problem
		counters::Counters # Counters of calls to functions like obj, grad, cons, of the problem
		nvar::Int64 # Number of variable of this problem (nvar_x + nvar_r)
		jres::Array{Int64, 1} # jres contains the indices of constraints with residues (feasible, not free and not linear)
		
	# Parameters for the objective function
		y::Array{Float64, 1}
		ρ::Float64
end

function NLCModel(nlp::AbstractNLPModel, mult::Array{Float64, 1}, penal::Float64, nlp_standard::Bool)::NLCModel
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
							x0   = vcat(nlp.meta.x0, ones(Float64, nvar_r)),
							lcon = nlp.meta.lcon, 
							ucon = nlp.meta.ucon, 
							name = nlp.meta.name
			)

		jres = Int64[]
		for i in 1:nlp.meta.ncon
			if !(i in nlp.meta.jinf) & !(i in nlp.meta.jfree) & !(i in nlp.meta.lin)
				push!(jres, i) 
			end
		end
	
	# Parameters
		y = mult
		ρ = penal

	# NLCModel created:
		return NLCModel(nlp, 
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


function NLPModels.obj(nlc::NLCModel, X::Array{Float64, 1})::Float64
	increment!(nlc, :neval_obj)
	if nlc.minimize
		return obj(nlc.nlp, X[1:nlc.nvar_x]) +
		       nlc.y' * X[nlc.nvar_x + 1 : end] +
			   0.5 * nlc.ρ * (norm(X[nlc.nvar_x + 1 : end], 2) ^ 2)
	
	else # argmax f(x) = argmin -f(x)
		nlc.minimize = true 
		return - obj(nlc.nlp, X[1:nlc.nvar_x]) +
			   nlc.y' * X[nlc.nvar_x + 1 : end] +
			   0.5 * nlc.ρ * (norm(X[nlc.nvar_x + 1 : end], 2) ^ 2)
	end
end


function NLPModels.grad(nlc::NLCModel, X::Array{Float64, 1}) ::Array{Float64, 1}
	increment!(nlc, :neval_grad)
	gx = vcat(grad(nlc.nlp, X[1:nlc.nvar_x]), nlc.ρ * X[nlc.nvar_x+1:end] + nlc.y)
	return gx
end

function NLPModels.grad!(nlc::NLCModel, X::Array{Float64, 1}, gx::Array{Float64, 1}) ::Array{Float64, 1}
	increment!(nlc, :neval_grad)
	if size(gx, 1) != nlc.nvar
		println("ERROR: wrong size of argument gx passed to grad! in NLCModel
				 gx should be of size " * string(nlc.nvar) * " but size " * string(size(gx)) * 
				 "given
				 Empty vector returned")
		return Float64[]
	end
	gx .= vcat(obj(nlc.nlp, X[1:nlc.nvar_x]), nlc.ρ * X[nlc.nvar_x+1:end] + nlc.y)
	return gx
end

# TODO (simple): sparse, pas matrice complète
# TODO : Check avec "Seul le lower triangle est retourné..."
function NLPModels.hess(nlc::NLCModel, X::Array{Float64, 1} ; obj_weight=1.0, y=zeros) ::Array{Float64, 2}
	increment!(nlc, :neval_hess)
	H = zeros(nlc.nvar, nlc.nvar)
	# Original information
		H[1:nlc.nvar_x, 1:nlc.nvar_x] = hess(nlc.nlp, X[1:nlc.nvar_x], obj_weight=obj_weight, y=y) # Original hessian
	
	# New information (due to residues)
		H[nlc.nvar_x+1:end, nlc.nvar_x+1:end] = H[nlc.nvar_x+1:end, nlc.nvar_x+1:end] + nlc.ρ * I # Added by residues (constant because of quadratic penalization) 
	
	# TODO: check there is no problem with the I::Uniformscaling operator (size...)
	return H
end

function NLPModels.hess_coord(nlc::NLCModel, X::Array{Float64, 1} ; obj_weight=1.0, y=zeros) #? type de retour ?
	increment!(nlc, :neval_hess)
	# Original information
		rows, cols, vals = hess_coord(nlc.nlp, X[1:nlc.nvar_x], obj_weight=obj_weight, y=y)
	
	# New information (due to residues)
		rows = vcat(rows, nlc.nvar_x+1:nlc.nvar)
		cols = vcat(cols, nlc.nvar_x+1:nlc.nvar)
		vals = vcat(vals, fill!(Vector{Float64}(undef, nlc.nvar_r), nlc.ρ)) # concatenate with a vector full of nlc.ρ
	return (rows, cols, vals)
end


function NLPModels.hprod(nlc::NLCModel, x::Array{Float64, 1}, v::Array{Float64, 1} ; obj_weight=1.0, y=zeros) ::Array{Float64, 1}
	increment!(nlc, :neval_hprod)
	# Test feasability
		if size(v, 1) != nlc.nvar
			println("ERROR: wrong size of argument v passed to jprod in NLCModel
					 gx should be of size " * string(nlc.nvar) * " but size " * string(size(v, 1)) * 
					 "given
					 Empty vector returned")
			return Float64[]
		end

	# Original information
		nlp_Hv = hprod(nlc.nlp, x, v[1:nlc.nvar_x], obj_weight=obj_weight, y=y)
	
	# New information (due to residues)
		Hv = vcat(nlp_Hv, nlc.ρ * v[nlc.nvar_x+1:end])
	
	return Hv
end



function NLPModels.cons(nlc::NLCModel, X::Array{Float64, 1}) ::Array{Float64, 1}
	increment!(nlc, :neval_cons)
	# Original information
		cx = cons(nlc.nlp, X[1:nlc.nvar_x]) # pre computation

	# New information (due to residues)
		for i in nlc.jres
			cx[i] += X[nlc.nvar_x + i] # residue for the i-th constraint (feasible, not free and not linear (not considered in this model))
		end
	
	return cx
end


function NLPModels.cons!(nlc::NLCModel, X::Array{Float64, 1}, cx::Array{Float64, 1}) ::Array{Float64, 1}
	increment!(nlc, :neval_cons)
	# Original information
		cx = cons(nlc.nlp, X[1:nlc.nvar_x]) # pre computation

	# New information (due to residues)
		for i in nlc.jres
			cx[i] += X[nlc.nvar_x + i] # residue for the i-th constraint (feasible, not free and not linear (not considered in this model))
		end

	return cx
end

# TODO (simple): return sparse, pas matrice complète
function NLPModels.jac(nlc::NLCModel, X::Array{Float64, 1}) ::Array{Float64, 2}
	increment!(nlc, :neval_jac)
	# Original information
		J = jac(nlc.nlp, X[1:nlc.nvar_x])

	# New information (due to residues)
		J = vcat(J, I) # residues part
		J = J[:, nlc.jres] # but some constraint don't have a residue, so we remove some
	return J
end

function NLPModels.jac_coord(nlc::NLCModel, X::Array{Float64, 1}) #? type de retour ?
	increment!(nlc, :neval_jac)
	# Original information
		rows, cols, vals = jac_coord(nlc.nlp, X[1:nlc.nvar_x])

	# New information (due to residues)
		rows = vcat(rows, 1:nlc.meta.ncon)
		cols = vcat(cols, nlc.nvar_x+1 : nlc.nvar_r)
		vals = vcat(vals, ones(Float64, nlc.nvar_r))
	
	return (rows, cols, vals)
end

function NLPModels.jprod!(nlc::NLCModel, X::Array{Float64, 1}, v::Array{Float64, 1}, Jv::Array{Float64, 1}) ::Array{Float64, 1}
	increment!(nlc, :neval_jprod)
	# Test feasability
		if size(v) != nlc.nvar
			println("ERROR: Wrong sizes of argument v passed to jprod\n
					Empty vector returned")
			return Float64[]
		end

	# Original information
		jprod!(nlc.nlp, X[1:nlc.nvar_x], v[1:nlc.nvar_x], Jv[1:nlc.nvar_x])
	
	# New information (due to residues)
		Jv = vcat(Jv, v[nlc.nvar_x+1:end])

	return Jv
end

function NLPModels.jtprod!(nlc::NLCModel, X::Array{Float64, 1}, v::Array{Float64, 1}, Jtv::Array{Float64, 1}) ::Array{Float64, 1}
	increment!(nlc, :neval_jtprod)
	# Test feasability
		if size(v) != nlc.ncon
			println("ERROR: Wrong sizes of argument v passed to jprod\n
					Empty vector returned")
			return Float64[]
		end

	# New information (due to residues)
		Resv = Float64[]
		for i in 1:nlc.ncon #? A tester !
			if i in nlc.jres
				push!(Resv, v[i]) 
			else
				push!(Resv, 0)
			end
		end

	# Original information
		Jv = jtprod(nlc.nlp, X[1:nlc.nvar_x], v) + Resv

	return Jv
end




function update_y(nlc, y_new)
	nlc.y = y_new
end

function update_ρ(nlc, ρ_new)
	nlc.ρ = ρ_new
end


