import NLPModels: increment!
using NLPModels
using LinearAlgebra
using SparseArrays
using Test

#** I. Model and constructor
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

	function NCLModel(nlp::AbstractNLPModel ; printing::Bool = false, ρ::Real = 1.0, )::NCLModel #TODO add rho y, ac val par defaut, type de retour, NLP/NCL...
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
					println("    NCLModel : added ", nvar_r, " residuals, at indices ", jres)
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
								y0   = nlp.meta.y0,
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
							ρ)
	end

#** II. Methods

	#** II.1 Objective function
		function NLPModels.obj(ncl::NCLModel, X::Vector{<:Real})::Real
			increment!(ncl, :neval_obj)
			obj_val = obj(ncl.nlp, X[1:ncl.nvar_x])
			if ncl.minimize
				obj_val +
				(ncl.y[1:ncl.nvar_r])' * X[ncl.nvar_x + 1 : ncl.nvar_x + ncl.nvar_r] +
				0.5 * ncl.ρ * (norm(X[ncl.nvar_x + 1 : ncl.nvar_x + ncl.nvar_r], 2) ^ 2)

			else # argmax f(x) = argmin -f(x)
				ncl.minimize = true 
				- obj_val +
				ncl.y' * X[ncl.nvar_x + 1 : ncl.nvar_x + ncl.nvar_r] +
				0.5 * ncl.ρ * (norm(X[ncl.nvar_x + 1 : ncl.nvar_x + ncl.nvar_r], 2) ^ 2)
			end
		end

#TODO grad_check
	#** II.2 Gradient of the objective function
		function NLPModels.grad(ncl::NCLModel, X::Vector{<:Real}) ::Vector{<:Real}
			increment!(ncl, :neval_grad)
			gx = vcat(grad(ncl.nlp, X[1:ncl.nvar_x]), ncl.ρ * X[ncl.nvar_x+1:end] + ncl.y)
			return gx
		end

		function NLPModels.grad!(ncl::NCLModel, X::Vector{<:Real}, gx::Vector{<:Real}) ::Vector{<:Real}
			increment!(ncl, :neval_grad)

			if length(gx) != ncl.nvar
				println("ERROR: wrong length of argument gx passed to grad! in NCLModel
						gx should be of length " * string(ncl.nvar) * " but length " * string(length(gx)) * 
						"given
						Empty vector returned")
				return <:Real[]
			end

			# Original information 
				grad!(ncl.nlp, X[1:ncl.nvar_x], gx) 
			
			# New information (due to residuals)
				gx[ncl.nvar_x + 1 : ncl.nvar_x + ncl.nvar_r] .= ncl.ρ * X[ncl.nvar_x + 1 : ncl.nvar_x + ncl.nvar_r] .+ ncl.y[1:ncl.nvar_r]

			return gx
		end

# TODO (simple): sparse du triangle inf, pas matrice complète
	#** II.3 Hessian of the Lagrangian
		function NLPModels.hess(ncl::NCLModel, X::Vector{<:Real} ; obj_weight=1.0, y=zeros) ::Matrix{<:Real}
			increment!(ncl, :neval_hess)
			H = zeros(ncl.nvar, ncl.nvar)
			# Original information
				H[1:ncl.nvar_x, 1:ncl.nvar_x] = hess(ncl.nlp, X[1:ncl.nvar_x], obj_weight=obj_weight, y=y) # Original hessian
			
			# New information (due to residuals)
				H[ncl.nvar_x+1:end, ncl.nvar_x+1:end] = H[ncl.nvar_x+1:end, ncl.nvar_x+1:end] + ncl.ρ * I # Added by residuals (constant because of quadratic penalization) 
		
			return H #?sparse(hess_coord(ncl, X, obj_weight=obj_weight, y=y)...)
		end

		function NLPModels.hess_coord(ncl::NCLModel, X::Vector{<:Real} ; obj_weight=1.0, y=zeros) ::Tuple{Vector{Int64},Vector{Int64},Vector{<:Real}}
			increment!(ncl, :neval_hess)
			# Original information
				hrows, hcols, hvals = hess_coord(ncl.nlp, X[1:ncl.nvar_x], obj_weight=obj_weight, y=y)
			
			# New information (due to residuals)
				append!(hrows, ncl.nvar_x+1:ncl.nvar)
				append!(hcols, ncl.nvar_x+1:ncl.nvar)
				append!(hvals, fill!(Vector{typeof(hvals[1])}(undef, ncl.nvar_r), ncl.ρ)) # concatenate with a vector full of ncl.ρ
			return (hrows, hcols, hvals)
		end

		function NLPModels.hess_structure(ncl::NCLModel) ::Tuple{Vector{Int64},Vector{Int64}}
			increment!(ncl, :neval_hess)
			# Original information
				hrows, hcols = hess_structure(ncl.nlp)
			
			# New information (due to residuals)
				append!(hrows, ncl.nvar_x+1:ncl.nvar)
				append!(hcols, ncl.nvar_x+1:ncl.nvar)
			return (hrows, hcols)
		end

		function NLPModels.hess_coord!(ncl::NCLModel, X::Vector{<:Real}, hrows::Vector{<:Int64}, hcols::Vector{<:Int64}, hvals::Vector{<:Real} ; obj_weight=1.0, y=zeros) ::Tuple{Vector{Int64},Vector{Int64},Vector{<:Real}}
			increment!(ncl, :neval_hess)
			#Pre computation
				len_hcols = length(hcols)
				orig_len = len_hcols - ncl.nvar_r

			# Original information
				hrows[1:orig_len], hcols[1:orig_len], hvals[1:orig_len] = hess_coord!(ncl.nlp, X[1:ncl.nvar_x], hrows[1:orig_len], hcols[1:orig_len], hvals[1:orig_len], obj_weight=obj_weight, y=y)
			
			# New information (due to residuals)
				hvals[orig_len + 1 : len_hcols] = fill!(Vector{typeof(hvals[1])}(undef, ncl.nvar_r), ncl.ρ) # a vector full of ncl.ρ
			return (hrows, hcols, hvals)
		end

		function NLPModels.hprod(ncl::NCLModel, X::Vector{<:Real}, v::Vector{<:Real} ; obj_weight=1.0, y=zeros) ::Vector{<:Real}
			increment!(ncl, :neval_hprod)
			# Test feasability
				if length(v) != ncl.nvar
					error("wrong length of argument v passed to hprod in NCLModel
						gx should be of length " * string(ncl.nvar) * " but length " * string(length(v)) * "given")
				end

			# Original information
				Hv = hprod(ncl.nlp, X[1:ncl.nvar_x], v[1:ncl.nvar_x], obj_weight=obj_weight, y=y)
			
			# New information (due to residuals)
				append!(Hv, ncl.ρ * v[ncl.nvar_x+1:end])
			
			return Hv
		end

		function NLPModels.hprod!(ncl::NCLModel, X::Vector{<:Real}, v::Vector{<:Real} , Hv::Vector{<:Real} ; obj_weight=1.0, y=zeros) ::Vector{<:Real}
			increment!(ncl, :neval_hprod)
			# Test feasability
				if length(v) != ncl.nvar
					error("wrong length of argument v passed to hprod in NCLModel
						gx should be of length " * string(ncl.nvar) * " but length " * string(length(v)) * "given")
				end

			# Original information
				Hv[1:ncl.nvar_x] = hprod!(ncl.nlp, X[1:ncl.nvar_x], v[1:ncl.nvar_x], Hv[1:ncl.nvar_x], obj_weight=obj_weight, y=y)
			
			# New information (due to residuals)
			Hv[ncl.nvar_x+1:end] = ncl.ρ * v[ncl.nvar_x+1:end]
			
			return Hv
		end

	#** II.4 Constraints
		function NLPModels.cons(ncl::NCLModel, X::Vector{<:Real}) ::Vector{<:Real}
			increment!(ncl, :neval_cons)
			# Original information
				cx = cons(ncl.nlp, X[1:ncl.nvar_x]) # pre computation

			# New information (due to residuals)
				cx[ncl.jres] += X[ncl.nvar_x+1:end] # residual for the i-th constraint (feasible, not free and not linear (not considered in this model))
			
			return cx
		end

		function NLPModels.cons!(ncl::NCLModel, X::Vector{<:Real}, cx::Vector{<:Real}) ::Vector{<:Real}
			increment!(ncl, :neval_cons)
			# Original information
				cons!(ncl.nlp, X[1:ncl.nvar_x], cx) # pre computation

			# New information (due to residuals)
				cx[ncl.jres] .+= X[ncl.nvar_x+1:end] # residual for the i-th constraint (feasible, not free and not linear (not considered in this model))

			return cx
		end

# TODO (simple): return sparse, pas matrice complète
#sparse(row col val)
	#** II.5 Jacobian of the constraints vector
		function NLPModels.jac(ncl::NCLModel, X::Vector{<:Real}) ::Matrix{<:Real}
			increment!(ncl, :neval_jac)
			# Original information
				J = jac(ncl.nlp, X[1:ncl.nvar_x])
				
			# New information (due to residuals)
				J = hcat(J, I) # residuals part
				J = J[1:end, vcat(1:ncl.nvar_x, ncl.jres .+ ncl.nvar_x)] # but some constraint don't have a residual, so we remove some
				
			return J
		end

		function NLPModels.jac_coord(ncl::NCLModel, X::Vector{<:Real}) ::Tuple{Vector{Int64},Vector{Int64},Vector{<:Real}}
			increment!(ncl, :neval_jac)
			# Original information
				jrows, jcols, jvals = jac_coord(ncl.nlp, X[1:ncl.nvar_x])

			# New information (due to residuals)
				append!(jrows, ncl.jres)
				append!(jcols, ncl.nvar_x+1 : ncl.nvar)
				append!(jvals, ones(typeof(jvals[1]), ncl.nvar_r))
			return (jrows, jcols, jvals)
		end

		function NLPModels.jac_coord!(ncl::NCLModel, X::Vector{<:Real}, jrows::Vector{<:Int64}, jcols::Vector{<:Int64}, jvals::Vector{<:Real}) ::Tuple{Vector{Int64},Vector{Int64},Vector{<:Real}}
			increment!(ncl, :neval_jac)
			#Pre computation
				len_jcols = length(jcols)
				orig_len = len_jcols - ncl.nvar_r

			# Original information
			jrows[1:orig_len], jcols[1:orig_len], jvals[1:orig_len] = jac_coord!(ncl.nlp, X[1:ncl.nvar_x], jrows[1:orig_len], jcols[1:orig_len], jvals[1:orig_len]) # we necessarily need the place for ncl.nvar_r ones in the value array

			# New information (due to residuals)
				jvals[orig_len + 1 : len_jcols] = ones(typeof(jvals[1]), ncl.nvar_r) # we assume length(jrows) = length(jcols) = length(jvals)

			return (jrows, jcols, jvals)
		end

		function NLPModels.jac_structure(ncl::NCLModel) ::Tuple{Vector{Int64},Vector{Int64}}
			increment!(ncl, :neval_jac)
			# Original information
				jrows, jcols = jac_structure(ncl.nlp)

			# New information (due to residuals) # ! If there is any problem, check the following :
				append!(jrows, ncl.jres) # ! important that jrows = [orignial_rows, residues_rows] for the jac_coor!() function
				append!(jcols, ncl.nvar_x+1 : ncl.nvar) # ! important that jcols = [orignial_cols, residues_cols] for the jac_coor!() function
			return jrows, jcols
		end

		function NLPModels.jprod(ncl::NCLModel, X::Vector{<:Real}, v::Vector{<:Real}) ::Vector{<:Real}
			increment!(ncl, :neval_jprod)
			# Test feasability
				if length(v) != ncl.nvar
					error("wrong sizes of argument v passed to jprod(ncl::NCLModel, X::Vector{<:Real}, v::Vector{<:Real}, Jv::Vector{<:Real}) ::Vector{<:Real}")
				end
				
			# Original information
				Jv = jprod(ncl.nlp, X[1:ncl.nvar_x], v[1:ncl.nvar_x])

			# New information (due to residuals)
				Resv = zeros(typeof(Jv[1,1]), ncl.nlp.meta.ncon)
				Resv[ncl.jres] = Resv[ncl.jres] + v[ncl.nvar_x+1:end]

			return Jv + Resv
		end

		function NLPModels.jprod!(ncl::NCLModel, X::Vector{<:Real}, v::Vector{<:Real}, Jv::Vector{<:Real}) ::Vector{<:Real}
			increment!(ncl, :neval_jprod)
			# Test feasability
				if length(v) != ncl.nvar
					error("wrong sizes of argument v passed to jprod!(ncl::NCLModel, X::Vector{<:Real}, v::Vector{<:Real}, Jv::Vector{<:Real}) ::Vector{<:Real}")
				end

			# Original information
				Jv .= jprod(ncl.nlp, X[1:ncl.nvar_x], v[1:ncl.nvar_x])
				
			# New information (due to residuals)
				Resv = zeros(typeof(Jv[1,1]), ncl.nlp.meta.ncon)
				Resv[ncl.jres] .+= v[ncl.nvar_x+1:end]
				Jv .+= Resv
				
			return Jv
		end

		function NLPModels.jtprod(ncl::NCLModel, X::Vector{<:Real}, v::Vector{<:Real}) ::Vector{<:Real}
			increment!(ncl, :neval_jtprod)
			# Test feasability
				if length(v) != ncl.nlp.meta.ncon
					error("wrong length of argument v passed to jtprod(ncl::NCLModel, X::Vector{<:Real}, v::Vector{<:Real}, Jtv::Vector{<:Real}) ::Vector{<:Real}")
				end

			# Original information
				Jv = jtprod(ncl.nlp, X[1:ncl.nvar_x], v)

			# New information (due to residuals)
				# v[ncl.jres]
			
			# Original information
				append!(Jv, v[ncl.jres])
			
			return Jv
		end

		function NLPModels.jtprod!(ncl::NCLModel, X::Vector{<:Real}, v::Vector{<:Real}, Jtv::Vector{<:Real}) ::Vector{<:Real}
			increment!(ncl, :neval_jtprod)
			# Test feasability
			if length(v) != ncl.nlp.meta.ncon
				error("wrong length of argument v passed to jtprod(ncl::NCLModel, X::Vector{<:Real}, v::Vector{<:Real}, Jtv::Vector{<:Real}) ::Vector{<:Real}")
			end

			# New information (due to residuals)
				Resv = v[ncl.jres]

			# Original information
				Jtv .= append!(jtprod(ncl.nlp, X[1:ncl.nvar_x], v), Resv)

			return Jtv
		end




