import NLPModels: increment!
using NLPModels
using LinearAlgebra
using Test


"""
Subtype of AbstractNLPModel, adapted to the NCL method. 
Keeps some informations from the original AbstractNLPModel, 
and creates a new problem, modifying 
	the objective function (sort of augmented lagrangian, without considering linear constraints) 
	the constraints (with residuals)
"""
mutable struct NCLModel <: AbstractNLPModel
	# Information about the original problem
		nlp::AbstractNLPModel # The original problem
		nvar_x::Int64 # Number of variable of the nlp problem
		nvar_r::Int64 # Number of residuals for the nlp problem (in fact nvar_r = length(nln), if there are no free/infeasible constraints)
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

function NCLModel(nlp::AbstractNLPModel ; printing = false::Bool)::NCLModel #TODO add rho y, ac val par defaut, type de retour, NLP/NCL...
	# * 0. printing
		if printing
			println("\nNLCModel called on " * nlp.meta.name)
		end
	
	
	# Information about the original problem
	#TODO modifier pour utiliser lin dans ADNLPModel
		if (nlp.meta.lin == Int[]) & (isa(nlp, ADNLPModel)) & (nlp.meta.name == "Unitary test problem")
			jres = [2, 4] 
			nvar_r = 2 # linear constraints are not considered here in the NCL method. 
		else
			nvar_r = nlp.meta.nnln # linear constraints are not considered here in the NCL method. 
			jres = nlp.meta.nln # copy, useless, but permits to use the unitary test problem computed
			
			if printing
				println("    NCLModel : jres = ", jres)
				println("    NCLModel : nvar_r = ", nvar_r)
			end
		end

		nvar_x = nlp.meta.nvar
		minimize = nlp.meta.minimize
		
	# Constant parameters
		nvar = nvar_x + nvar_r
		meta = NLPModelMeta(nvar ;
							lvar = vcat(nlp.meta.lvar, -Inf * ones(nvar_r)), # No bounds upon residuals
							uvar = vcat(nlp.meta.uvar, Inf * ones(nvar_r)),
							x0   = vcat(nlp.meta.x0, ones(typeof(nlp.meta.x0[1]), nvar_r)),
							y0   = zeros(typeof(nlp.meta.x0[1]), nlp.meta.ncon), # TODO celui de nlp meta
							name = nlp.meta.name * " (NCL subproblem)",
							nnzj = nlp.meta.nnzj + nvar_r, # we add nonzeros because of residuals
							nnzh = nlp.meta.nnzh + nvar_r,
							ncon = nlp.meta.ncon,
							lcon = nlp.meta.lcon,
							ucon = nlp.meta.ucon
							)

		if nlp.meta.jinf != Int64[]
			error("argument problem passed to NCLModel with constraint " * string(nlp.meta.jinf) * " infeasible")
		end
		if nlp.meta.jfree != Int64[]
			error("argument problem passed to NCLModel with constraint " * string(nlp.meta.jfree) * " free")
		end
	
	# Parameters
		#if length(mult) != nvar_r # ? Utile ?
		#	y = zeros(typeof(nlp.meta.x0[1]), nvar_r)
		#else
		#	y = mult
		#end
		#ρ = penal

	# NCLModel created:
		return NCLModel(nlp, 
						nvar_x, 
						nvar_r, 
						minimize,	
						jres,		
						meta, 
						Counters(), 
						nvar,
						zeros(typeof(nlp.meta.x0[1]), nvar_r), 
						1.0)
end


function NLPModels.obj(nlc::NCLModel, X::Vector{<:Real})::Real
	increment!(nlc, :neval_obj)
	if nlc.minimize
		if nlc.nvar_r == 0 # little test to avoid []' * [] #TODO: simplifier, inutile
			return obj(nlc.nlp, X[1:nlc.nvar_x])
		else
			return obj(nlc.nlp, X[1:nlc.nvar_x]) +
				(nlc.y[1:nlc.nvar_r])' * X[nlc.nvar_x + 1 : nlc.nvar_x + nlc.nvar_r] +
				0.5 * nlc.ρ * (norm(X[nlc.nvar_x + 1 : nlc.nvar_x + nlc.nvar_r], 2) ^ 2)
		end

	else # argmax f(x) = argmin -f(x)
		nlc.minimize = true 
		if nlc.nvar_r == 0 
			return - obj(nlc.nlp, X[1:nlc.nvar_x])
		else
			return - obj(nlc.nlp, X[1:nlc.nvar_x]) +
				nlc.y' * X[nlc.nvar_x + 1 : nlc.nvar_x + nlc.nvar_r] +
				0.5 * nlc.ρ * (norm(X[nlc.nvar_x + 1 : nlc.nvar_x + nlc.nvar_r], 2) ^ 2)
		end
	end
end



function NLPModels.grad(nlc::NCLModel, X::Vector{<:Real}) ::Vector{<:Real}
	increment!(nlc, :neval_grad)
	gx = vcat(grad(nlc.nlp, X[1:nlc.nvar_x]), nlc.ρ * X[nlc.nvar_x+1:end] + nlc.y)
	return gx
end
#TODO grad_check
function NLPModels.grad!(nlc::NCLModel, X::Vector{<:Real}, gx::Vector{<:Real}) ::Vector{<:Real}
	increment!(nlc, :neval_grad)

	if length(gx) != nlc.nvar
		println("ERROR: wrong length of argument gx passed to grad! in NCLModel
				 gx should be of length " * string(nlc.nvar) * " but length " * string(length(gx)) * 
				 "given
				 Empty vector returned")
		return <:Real[]
	end

	# Original information 
		grad!(nlc.nlp, X[1:nlc.nvar_x], gx) 
	
	# New information (due to residuals) # TODO residual
		gx[nlc.nvar_x + 1 : nlc.nvar_x + nlc.nvar_r] .= nlc.ρ * X[nlc.nvar_x + 1 : nlc.nvar_x + nlc.nvar_r] .+ nlc.y[1:nlc.nvar_r]

	return gx
end



# TODO (simple): sparse du triangle inf, pas matrice complète
function NLPModels.hess(nlc::NCLModel, X::Vector{<:Real} ; obj_weight=1.0, y=zeros) ::Matrix{<:Real}
	increment!(nlc, :neval_hess)
	H = zeros(nlc.nvar, nlc.nvar)
	# Original information
		H[1:nlc.nvar_x, 1:nlc.nvar_x] = hess(nlc.nlp, X[1:nlc.nvar_x], obj_weight=obj_weight, y=y) # Original hessian
	
	# New information (due to residuals)
		H[nlc.nvar_x+1:end, nlc.nvar_x+1:end] = H[nlc.nvar_x+1:end, nlc.nvar_x+1:end] + nlc.ρ * I # Added by residuals (constant because of quadratic penalization) 
	
	return H
end

function NLPModels.hess_coord(nlc::NCLModel, X::Vector{<:Real} ; obj_weight=1.0, y=zeros) ::Tuple{Vector{Int64},Vector{Int64},Vector{<:Real}}
	increment!(nlc, :neval_hess)
	# Original information
		hrows, hcols, hvals = hess_coord(nlc.nlp, X[1:nlc.nvar_x], obj_weight=obj_weight, y=y)
	
	# New information (due to residuals)
		append!(hrows, nlc.nvar_x+1:nlc.nvar)
		append!(hcols, nlc.nvar_x+1:nlc.nvar)
		append!(hvals, fill!(Vector{typeof(hvals[1])}(undef, nlc.nvar_r), nlc.ρ)) # concatenate with a vector full of nlc.ρ
	return (hrows, hcols, hvals)
end

function NLPModels.hess_structure(nlc::NCLModel) ::Tuple{Vector{Int64},Vector{Int64}}
	increment!(nlc, :neval_hess)
	# Original information
		hrows, hcols = hess_structure(nlc.nlp)
	
	# New information (due to residuals)
		append!(hrows, nlc.nvar_x+1:nlc.nvar)
		append!(hcols, nlc.nvar_x+1:nlc.nvar)
	return (hrows, hcols)
end

function NLPModels.hess_coord!(nlc::NCLModel, X::Vector{<:Real}, hrows::Vector{<:Int64}, hcols::Vector{<:Int64}, hvals::Vector{<:Real} ; obj_weight=1.0, y=zeros) ::Tuple{Vector{Int64},Vector{Int64},Vector{<:Real}}
	increment!(nlc, :neval_hess)
	#Pre computation
		len_hcols = length(hcols)
		orig_len = len_hcols - nlc.nvar_r

	# Original information
		hrows[1:orig_len], hcols[1:orig_len], hvals[1:orig_len] = hess_coord!(nlc.nlp, X[1:nlc.nvar_x], hrows[1:orig_len], hcols[1:orig_len], hvals[1:orig_len], obj_weight=obj_weight, y=y)
	
	# New information (due to residuals)
		hvals[orig_len + 1 : len_hcols] = fill!(Vector{typeof(hvals[1])}(undef, nlc.nvar_r), nlc.ρ) # a vector full of nlc.ρ
	return (hrows, hcols, hvals)
end

function NLPModels.hprod(nlc::NCLModel, X::Vector{<:Real}, v::Vector{<:Real} ; obj_weight=1.0, y=zeros) ::Vector{<:Real}
	increment!(nlc, :neval_hprod)
	# Test feasability
		if length(v) != nlc.nvar
			error("wrong length of argument v passed to hprod in NCLModel
				   gx should be of length " * string(nlc.nvar) * " but length " * string(length(v)) * "given")
		end

	# Original information
		Hv = hprod(nlc.nlp, X[1:nlc.nvar_x], v[1:nlc.nvar_x], obj_weight=obj_weight, y=y)
	
	# New information (due to residuals)
		append!(Hv, nlc.ρ * v[nlc.nvar_x+1:end])
	
	return Hv
end

function NLPModels.hprod!(nlc::NCLModel, X::Vector{<:Real}, v::Vector{<:Real} , Hv::Vector{<:Real} ; obj_weight=1.0, y=zeros) ::Vector{<:Real}
	increment!(nlc, :neval_hprod)
	# Test feasability
		if length(v) != nlc.nvar
			error("wrong length of argument v passed to hprod in NCLModel
				   gx should be of length " * string(nlc.nvar) * " but length " * string(length(v)) * "given")
		end

	# Original information
		Hv[1:nlc.nvar_x] = hprod!(nlc.nlp, X[1:nlc.nvar_x], v[1:nlc.nvar_x], Hv[1:nlc.nvar_x], obj_weight=obj_weight, y=y)
	
	# New information (due to residuals)
	Hv[nlc.nvar_x+1:end] = nlc.ρ * v[nlc.nvar_x+1:end]
	
	return Hv
end


function NLPModels.cons(nlc::NCLModel, X::Vector{<:Real}) ::Vector{<:Real}
	increment!(nlc, :neval_cons)
	# Original information
		cx = cons(nlc.nlp, X[1:nlc.nvar_x]) # pre computation

	# New information (due to residuals)
		cx[nlc.jres] += X[nlc.nvar_x+1:end] # residual for the i-th constraint (feasible, not free and not linear (not considered in this model))
	
	return cx
end

function NLPModels.cons!(nlc::NCLModel, X::Vector{<:Real}, cx::Vector{<:Real}) ::Vector{<:Real}
	increment!(nlc, :neval_cons)
	# Original information
		cons!(nlc.nlp, X[1:nlc.nvar_x], cx) # pre computation

	# New information (due to residuals)
		cx[nlc.jres] .+= X[nlc.nvar_x+1:end] # residual for the i-th constraint (feasible, not free and not linear (not considered in this model))

	return cx
end

# TODO (simple): return sparse, pas matrice complète
#sparse(row col val)
function NLPModels.jac(nlc::NCLModel, X::Vector{<:Real}) ::Matrix{<:Real}
	increment!(nlc, :neval_jac)
	# Original information
		J = jac(nlc.nlp, X[1:nlc.nvar_x])
		
	# New information (due to residuals)
		J = hcat(J, I) # residuals part
		J = J[1:end, vcat(1:nlc.nvar_x, nlc.jres .+ nlc.nvar_x)] # but some constraint don't have a residual, so we remove some
		
	return J
end

function NLPModels.jac_coord(nlc::NCLModel, X::Vector{<:Real}) ::Tuple{Vector{Int64},Vector{Int64},Vector{<:Real}}
	increment!(nlc, :neval_jac)
	# Original information
		jrows, jcols, jvals = jac_coord(nlc.nlp, X[1:nlc.nvar_x])

	# New information (due to residuals)
		append!(jrows, nlc.jres)
		append!(jcols, nlc.nvar_x+1 : nlc.nvar)
		append!(jvals, ones(typeof(jvals[1]), nlc.nvar_r))
	return (jrows, jcols, jvals)
end

function NLPModels.jac_coord!(nlc::NCLModel, X::Vector{<:Real}, jrows::Vector{<:Int64}, jcols::Vector{<:Int64}, jvals::Vector{<:Real}) ::Tuple{Vector{Int64},Vector{Int64},Vector{<:Real}}
	increment!(nlc, :neval_jac)
	#Pre computation
		len_jcols = length(jcols)
		orig_len = len_jcols - nlc.nvar_r

	# Original information
	jrows[1:orig_len], jcols[1:orig_len], jvals[1:orig_len] = jac_coord!(nlc.nlp, X[1:nlc.nvar_x], jrows[1:orig_len], jcols[1:orig_len], jvals[1:orig_len]) # we necessarily need the place for nlc.nvar_r ones in the value array

	# New information (due to residuals)
		jvals[orig_len + 1 : len_jcols] = ones(typeof(jvals[1]), nlc.nvar_r) # we assume length(jrows) = length(jcols) = length(jvals)

	return (jrows, jcols, jvals)
end


function NLPModels.jac_structure(nlc::NCLModel) ::Tuple{Vector{Int64},Vector{Int64}}
	increment!(nlc, :neval_jac)
	# Original information
		jrows, jcols = jac_structure(nlc.nlp)

	# New information (due to residuals) # ! If there is any problem, check the following :
		append!(jrows, nlc.jres) # ! important that jrows = [orignial_rows, residues_rows] for the jac_coor!() function
		append!(jcols, nlc.nvar_x+1 : nlc.nvar) # ! important that jcols = [orignial_cols, residues_cols] for the jac_coor!() function
	return jrows, jcols
end

function NLPModels.jprod(nlc::NCLModel, X::Vector{<:Real}, v::Vector{<:Real}) ::Vector{<:Real}
	increment!(nlc, :neval_jprod)
	# Test feasability
		if length(v) != nlc.nvar
			error("wrong sizes of argument v passed to jprod(nlc::NCLModel, X::Vector{<:Real}, v::Vector{<:Real}, Jv::Vector{<:Real}) ::Vector{<:Real}")
		end
		
	# Original information
		Jv = jprod(nlc.nlp, X[1:nlc.nvar_x], v[1:nlc.nvar_x])

	# New information (due to residuals)
		Resv = zeros(typeof(Jv[1,1]), nlc.nlp.meta.ncon)
		Resv[nlc.jres] = Resv[nlc.jres] + v[nlc.nvar_x+1:end]

	return Jv + Resv
end

function NLPModels.jprod!(nlc::NCLModel, X::Vector{<:Real}, v::Vector{<:Real}, Jv::Vector{<:Real}) ::Vector{<:Real}
	increment!(nlc, :neval_jprod)
	# Test feasability
		if length(v) != nlc.nvar
			error("wrong sizes of argument v passed to jprod!(nlc::NCLModel, X::Vector{<:Real}, v::Vector{<:Real}, Jv::Vector{<:Real}) ::Vector{<:Real}")
		end

	# Original information
		Jv .= jprod(nlc.nlp, X[1:nlc.nvar_x], v[1:nlc.nvar_x])
		
	# New information (due to residuals)
		Resv = zeros(typeof(Jv[1,1]), nlc.nlp.meta.ncon)
		Resv[nlc.jres] .+= v[nlc.nvar_x+1:end]
		Jv .+= Resv
		
	return Jv
end

function NLPModels.jtprod(nlc::NCLModel, X::Vector{<:Real}, v::Vector{<:Real}) ::Vector{<:Real}
	increment!(nlc, :neval_jtprod)
	# Test feasability
		if length(v) != nlc.nlp.meta.ncon
			error("wrong length of argument v passed to jtprod(nlc::NCLModel, X::Vector{<:Real}, v::Vector{<:Real}, Jtv::Vector{<:Real}) ::Vector{<:Real}")
		end

	# Original information
		Jv = jtprod(nlc.nlp, X[1:nlc.nvar_x], v)

	# New information (due to residuals)
		# v[nlc.jres]
	
	# Original information
		append!(Jv, v[nlc.jres])
	
	return Jv
end

function NLPModels.jtprod!(nlc::NCLModel, X::Vector{<:Real}, v::Vector{<:Real}, Jtv::Vector{<:Real}) ::Vector{<:Real}
	increment!(nlc, :neval_jtprod)
	# Test feasability
	if length(v) != nlc.nlp.meta.ncon
		error("wrong length of argument v passed to jtprod(nlc::NCLModel, X::Vector{<:Real}, v::Vector{<:Real}, Jtv::Vector{<:Real}) ::Vector{<:Real}")
	end

	# New information (due to residuals)
		Resv = v[nlc.jres]

	# Original information
		Jtv .= append!(jtprod(nlc.nlp, X[1:nlc.nvar_x], v), Resv)

	return Jtv
end




