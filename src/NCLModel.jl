import NLPModels: increment!
using NLPModels
using LinearAlgebra
using SparseArrays
using Test
using Printf

######### TODO #########
######### TODO #########
######### TODO #########
	# TODO grad_check
	# TODO (simple): terminer la fonction print et documentation
######### TODO #########
######### TODO #########
######### TODO #########






#** I. Model and constructor
	"""
	######################
	NCLModel documentation
		Subtype of AbstractNLPModel, adapted to the NCL method. 
		Keeps some informations from the original AbstractNLPModel, 
		and creates a new problem, modifying 
			the objective function (sort of augmented lagrangian, with residuals instead of constraints) 
			the constraints (with residuals)

		Process is as follows

			(nlp) | min_{x} f(x)								         | min_{x,r} f(x) + λ' * r + ρ * ||r||²		     	(λ and ρ are parameters)
				  | subject to lvar <= x <= uvar		becomes: (ncl)   | subject to lvar <= x <= uvar, -Inf <= r <= Inf
				  | 		   lcon <= c(x) <= ucon				         | 			lcon <= c(x) + r <= ucon

	######################
	"""
	mutable struct NCLModel <: AbstractNLPModel
		#* I. Information about the residuals
			nlp::AbstractNLPModel # The original problem
			nvar_x::Int64 # Number of variable of the nlp problem
			nvar_r::Int64 # Number of residuals for the nlp problem (in fact nvar_r = length(nln), if there are no free/infeasible constraints)
			minimize::Bool # true if the aim of the problem is to minimize, false otherwise
			res_lin_cons::Bool # Boolean to chose if you put residuals upon linear constraints (true) or not

		#* II. Constant parameters
			meta::AbstractNLPModelMeta # Informations for this problem
			counters::Counters # Counters of calls to functions like obj, grad, cons, of the problem
			nvar::Int64 # Number of variable of this problem (nvar_x + nvar_r)
			
		#* III. Parameters for the objective function
			y::Vector{<:Float64} # Multipliers for the nlp problem, used in the lagrangian
			ρ::Float64 # Penalization of the simili lagrangian
	end



	"""
	######################
	NCLModel documentation
		Creates the NCL problem associated to the nlp in argument. 
	######################
	"""
	function NCLModel(nlp::AbstractNLPModel ; 															# Initial model
						print_level::Int64 = 0, 															# Little informationwarnings about the model created
						res_val_init::Float64 = 0., 														# Initial value for residuals
						res_lin_cons::Bool = false, 														# Choose if you want residuals upon linear constraints or not
						ρ::Float64 = 1., 																	# Initial penalization
						y = res_lin_cons ? zeros(Float64, nlp.meta.ncon) : zeros(Float64, nlp.meta.nnln)	# Initial multiplier, depending on the number of residuals considered
					 ) ::AbstractNLPModel 																# Return an AbstractNLPModel, because if there is no residuals to add, it is better to return the original NLP problem. A warning is displayed in this case

		#* 0. Printing
			if print_level >= 1
				println("\nNLCModel called on " * nlp.meta.name)

				
				if print_level >= 2 
					if res_lin_cons
						println("    Residuals on linear constraints, (set res_lin_cons to false, if you want to consider only non linear constraints)")
					else
						println("    No residuals on linear constraints, only non linear are considered (set res_lin_cons to true, if you want to consider linear constraints as well)")
					end
				end
			end
		
		
		#* I. First tests
			#* I.1 Need to create a NCLModel ?
				if (nlp.meta.ncon == 0) # No need to create an NCLModel, because it is an unconstrained problem or it doesn't have non linear constraints
					@warn("The nlp problem given was unconstrained, so it was returned without modification.")
					return(nlp)
				end

				if ((nlp.meta.nnln == 0) & !res_lin_cons) # No need to create an NCLModel, because we don't put residuals upon linear constraints (and there are not  any non linear constraint)
					@warn("The nlp problem given was linearly constrained, so it was returned without modification. \nConsider setting res_lin_cons to true if you want residuals upon linear constraints.")
					return(nlp)
				end


			#* I.2 Residuals treatment
				nvar_r = res_lin_cons ? nlp.meta.ncon : nlp.meta.nnln

				if print_level >= 2 
					println("    NCLModel : added ", nvar_r, " residuals")
				end


		#* II. Meta field
			nvar_x = nlp.meta.nvar
			minimize = nlp.meta.minimize
			nvar = nvar_x + nvar_r
			meta = NLPModelMeta(nvar ;
								lvar = vcat(nlp.meta.lvar, fill!(Vector{Float64}(undef, nvar_r), -Inf)), # No bounds upon residuals
								uvar = vcat(nlp.meta.uvar, fill!(Vector{Float64}(undef, nvar_r), Inf)),
								x0   = vcat(nlp.meta.x0, fill!(Vector{Float64}(undef, nvar_r), res_val_init)),
								y0   = nlp.meta.y0,
								name = nlp.meta.name * " (NCL subproblem)",
								nnzj = nlp.meta.nnzj + nvar_r, # we add nonzeros because of residuals
								nnzh = nlp.meta.nnzh + nvar_r,
								ncon = nlp.meta.ncon,
								lcon = nlp.meta.lcon,
								ucon = nlp.meta.ucon,
								)

			if nlp.meta.jinf != Int64[]
				error("argument problem passed to NCLModel with constraint " * string(nlp.meta.jinf) * " infeasible")
			end
			if nlp.meta.jfree != Int64[]
				error("argument problem passed to NCLModel with constraint " * string(nlp.meta.jfree) * " free")
			end

			if nlp.meta.iinf != Int64[]
				error("argument problem passed to NCLModel with bound constraint " * string(nlp.meta.iinf) * " infeasible")
			end


		#* III. NCLModel created:
			return NCLModel(nlp, 
							nvar_x, 
							nvar_r, 
							minimize,
							res_lin_cons,	
							meta, 
							Counters(), 
							nvar,
							y, 
							ρ)
	end





#** II. Methods

	#** II.1 Objective function
		function NLPModels.obj(ncl::NCLModel, X::Vector{<:Float64})::Float64
			increment!(ncl, :neval_obj)
			obj_val = obj(ncl.nlp, X[1:ncl.nvar_x])
			obj_res = (ncl.y[1:ncl.nvar_r])' * X[ncl.nvar_x + 1 : ncl.nvar_x + ncl.nvar_r] +
					   0.5 * ncl.ρ * reduce(+, X[i] * X[i] for i in ncl.nvar_x + 1 : ncl.nvar_x + ncl.nvar_r ; init=0)
					  
			if ncl.minimize
				return obj_val + obj_res
			else # argmax f(x) = argmin -f(x)
				ncl.minimize = true 
				- obj_val + obj_res
			end
		end



	#** II.2 Gradient of the objective function
		function NLPModels.grad(ncl::NCLModel, X::Vector{<:Float64}) ::Vector{<:Float64}
			increment!(ncl, :neval_grad)
			gx = vcat(grad(ncl.nlp, X[1:ncl.nvar_x]), ncl.ρ * X[ncl.nvar_x+1:end] + ncl.y)
			return gx
		end



		function NLPModels.grad!(ncl::NCLModel, X::Vector{<:Float64}, gx::Vector{<:Float64}) ::Vector{<:Float64}
			increment!(ncl, :neval_grad)

			if length(gx) != ncl.nvar
				error("wrong length of argument gx passed to grad! in NCLModel
					   gx should be of length " * string(ncl.nvar) * " but length " * string(length(gx)) * "given")
			end

			# Original information 
				grad!(ncl.nlp, X[1:ncl.nvar_x], gx) 
			
			# New information (due to residuals)
				gx[ncl.nvar_x + 1 : ncl.nvar_x + ncl.nvar_r] .= ncl.ρ * X[ncl.nvar_x + 1 : ncl.nvar_x + ncl.nvar_r] .+ ncl.y[1:ncl.nvar_r]

			return gx
		end




	#** II.3 Hessian of the Lagrangian
		function NLPModels.hess(ncl::NCLModel, X::Vector{<:Float64} ; obj_weight=1.0, y=zeros) ::SparseMatrixCSC{<:Float64, Int64}
			increment!(ncl, :neval_hess)
			
			H = sparse(hess_coord(ncl, X ; obj_weight=obj_weight, y=y)[1], hess_coord(ncl, X ; obj_weight=obj_weight, y=y)[2], hess_coord(ncl, X ; obj_weight=obj_weight, y=y)[3])

			return H
		end


		function NLPModels.hess_coord(ncl::NCLModel, X::Vector{<:Float64} ; obj_weight=1.0, y=zeros) ::Tuple{Vector{Int64},Vector{Int64},Vector{<:Float64}}
			increment!(ncl, :neval_hess)
			# Original information
				hrows, hcols, hvals = hess_coord(ncl.nlp, X[1:ncl.nvar_x], obj_weight=obj_weight, y=y)
			
			# New information (due to residuals)
				append!(hrows, ncl.nvar_x+1:ncl.nvar)
				append!(hcols, ncl.nvar_x+1:ncl.nvar)
				append!(hvals, fill!(Vector{typeof(hvals[1])}(undef, ncl.nvar_r), ncl.ρ)) # concatenate with a vector full of ncl.ρ
			return (hrows, hcols, hvals)
		end


		function NLPModels.hess_coord!(ncl::NCLModel, X::Vector{<:Float64}, hrows::Vector{<:Int64}, hcols::Vector{<:Int64}, hvals::Vector{<:Float64} ; obj_weight=1.0, y=zeros) ::Tuple{Vector{Int64},Vector{Int64},Vector{<:Float64}}
			increment!(ncl, :neval_hess)
			#Pre computation
				len_hcols = length(hcols)
				orig_len = len_hcols - ncl.nvar_r

			# Original information
				hvals[1:orig_len] .= hess_coord!(ncl.nlp, X[1:ncl.nvar_x], hrows[1:orig_len], hcols[1:orig_len], hvals[1:orig_len], obj_weight=obj_weight, y=y)[3]
			
			# New information (due to residuals)
				hvals[orig_len + 1 : len_hcols] .= fill!(Vector{typeof(hvals[1])}(undef, ncl.nvar_r), ncl.ρ) # a vector full of ncl.ρ

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


		function NLPModels.hprod(ncl::NCLModel, X::Vector{<:Float64}, v::Vector{<:Float64} ; obj_weight=1.0, y=zeros) ::Vector{<:Float64}
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


		function NLPModels.hprod!(ncl::NCLModel, X::Vector{<:Float64}, v::Vector{<:Float64} , Hv::Vector{<:Float64} ; obj_weight::Float64=1.0, y=zeros) ::Vector{<:Float64}
			increment!(ncl, :neval_hprod)
			# Test feasability
				if length(v) != ncl.nvar
					error("wrong length of argument v passed to hprod in NCLModel
						gx should be of length " * string(ncl.nvar) * " but length " * string(length(v)) * "given")
				end

			# Original information
				Hv[1:ncl.nvar_x] .= hprod!(ncl.nlp, X[1:ncl.nvar_x], v[1:ncl.nvar_x], Hv[1:ncl.nvar_x], obj_weight=obj_weight, y=y)
			
			# New information (due to residuals)
			Hv[ncl.nvar_x+1:end] .= ncl.ρ * v[ncl.nvar_x+1:end]
			
			return Hv
		end





	#** II.4 Constraints
		function NLPModels.cons(ncl::NCLModel, X::Vector{<:Float64}) ::Vector{<:Float64}
			increment!(ncl, :neval_cons)
			# Original information
				cx = cons(ncl.nlp, X[1:ncl.nvar_x]) # pre computation

			# New information (due to residuals)
				if ncl.res_lin_cons
					cx += X[ncl.nvar_x+1:end] # a constraint on every residual
				else
					cx[ncl.nlp.meta.nln] += X[ncl.nvar_x+1:end] # residual for the i-th constraint (feasible, not free and not linear (not considered in this model))
				end
			
			return cx
		end


		function NLPModels.cons!(ncl::NCLModel, X::Vector{<:Float64}, cx::Vector{<:Float64}) ::Vector{<:Float64}
			increment!(ncl, :neval_cons)
			# Original information
				cons!(ncl.nlp, X[1:ncl.nvar_x], cx) # pre computation

			# New information (due to residuals)
				if ncl.res_lin_cons
					cx .+= X[ncl.nvar_x+1:end]
				else
					cx[ncl.nlp.meta.nln] .+= X[ncl.nvar_x+1:end] # residual for the i-th constraint (feasible, not free and not linear (not considered in this model))
				end

			return cx
		end



	#** II.5 Jacobian of the constraints vector
		function NLPModels.jac(ncl::NCLModel, X::Vector{<:Float64}) ::SparseMatrixCSC{<:Float64, Int64}
			increment!(ncl, :neval_jac)
				
			J = sparse(jac_coord(ncl, X)[1], jac_coord(ncl, X)[2], jac_coord(ncl, X)[3])

			return J
		end


		function NLPModels.jac_coord(ncl::NCLModel, X::Vector{<:Float64}) ::Tuple{Vector{Int64},Vector{Int64},Vector{<:Float64}}
			increment!(ncl, :neval_jac)
			# Original information
				jrows, jcols, jvals = jac_coord(ncl.nlp, X[1:ncl.nvar_x])

			# New information (due to residuals)
				if ncl.res_lin_cons
					append!(jrows, 1:ncl.meta.ncon)
				else
					append!(jrows, ncl.nlp.meta.nln)
				end
				append!(jcols, ncl.nvar_x+1 : ncl.nvar)
				append!(jvals, ones(typeof(jvals[1]), ncl.nvar_r))
			return (jrows, jcols, jvals)
		end


		function NLPModels.jac_coord!(ncl::NCLModel, X::Vector{<:Float64}, jrows::Vector{<:Int64}, jcols::Vector{<:Int64}, jvals::Vector{<:Float64}) ::Tuple{Vector{Int64},Vector{Int64},Vector{<:Float64}}
			increment!(ncl, :neval_jac)

			#Pre computation
				len_jcols = length(jcols)
				orig_len = len_jcols - ncl.nvar_r

			# Test feasability
				if length(jvals) != len_jcols
					error("wrong sizes of argument jvals passed to jac_coord!(ncl::NCLModel, X::Vector{<:Float64}, jrows::Vector{<:Int64}, jcols::Vector{<:Int64}, jvals::Vector{<:Float64}) ::Tuple{Vector{Int64},Vector{Int64},Vector{<:Float64}}")
				end

			# Original informations
				jvals[1:orig_len] .= jac_coord!(ncl.nlp, X[1:ncl.nvar_x], jrows[1:orig_len], jcols[1:orig_len], jvals[1:orig_len])[3] # we necessarily need the place for ncl.nvar_r ones in the value array

			# New information (due to residuals)
				jvals[orig_len + 1 : len_jcols] = ones(typeof(jvals[1]), ncl.nvar_r) # we assume length(jrows) = length(jcols) = length(jvals)

			return (jrows, jcols, jvals)
		end


		function NLPModels.jac_structure(ncl::NCLModel) ::Tuple{Vector{Int64},Vector{Int64}}
			increment!(ncl, :neval_jac)
			# Original information
				jrows, jcols = jac_structure(ncl.nlp)

			# New information (due to residuals)
				if ncl.res_lin_cons
					append!(jrows, 1:ncl.meta.ncon)
				else
					append!(jrows, ncl.nlp.meta.nln)
				end
				append!(jcols, ncl.nvar_x+1 : ncl.nvar)
			return jrows, jcols
		end


		function NLPModels.jprod(ncl::NCLModel, X::Vector{<:Float64}, v::Vector{<:Float64}) ::Vector{<:Float64}
			increment!(ncl, :neval_jprod)
			# Test feasability
				if length(v) != ncl.nvar
					error("wrong sizes of argument v passed to jprod(ncl::NCLModel, X::Vector{<:Float64}, v::Vector{<:Float64}, Jv::Vector{<:Float64}) ::Vector{<:Float64}")
				end
				
			# Original information
				Jv = jprod(ncl.nlp, X[1:ncl.nvar_x], v[1:ncl.nvar_x])

			# New information (due to residuals)
				Resv = zeros(typeof(Jv[1,1]), ncl.meta.ncon)
				if ncl.res_lin_cons
					Resv += v[ncl.nvar_x+1:end]
				else
					Resv[ncl.nlp.meta.nln] += v[ncl.nvar_x+1:end]
				end

			return Jv + Resv
		end


		function NLPModels.jprod!(ncl::NCLModel, X::Vector{<:Float64}, v::Vector{<:Float64}, Jv::Vector{<:Float64}) ::Vector{<:Float64}
			increment!(ncl, :neval_jprod)
			# Test feasability
				if length(v) != ncl.nvar
					error("wrong sizes of argument v passed to jprod!(ncl::NCLModel, X::Vector{<:Float64}, v::Vector{<:Float64}, Jv::Vector{<:Float64}) ::Vector{<:Float64}")
				end

			# Original information
				jprod!(ncl.nlp, X[1:ncl.nvar_x], v[1:ncl.nvar_x], Jv)
				
			# New information (due to residuals)
				Resv = zeros(typeof(Jv[1,1]), ncl.meta.ncon)
				if ncl.res_lin_cons
					Resv = v[ncl.nvar_x+1:end]
				else
					Resv[ncl.nlp.meta.nln] += v[ncl.nvar_x+1:end]
				end
				Jv .+= Resv
				
			return Jv
		end


		function NLPModels.jtprod(ncl::NCLModel, X::Vector{<:Float64}, v::Vector{<:Float64}) ::Vector{<:Float64}
			increment!(ncl, :neval_jtprod)
			# Test feasability
				if length(v) != ncl.meta.ncon
					error("wrong length of argument v passed to jtprod(ncl::NCLModel, X::Vector{<:Float64}, v::Vector{<:Float64}, Jtv::Vector{<:Float64}) ::Vector{<:Float64}")
				end

			# Original information
				Jv = jtprod(ncl.nlp, X[1:ncl.nvar_x], v)

			# New information (due to residuals)
				if ncl.res_lin_cons
					append!(Jv, v)
				else
					append!(Jv, v[ncl.nlp.meta.nln])
				end
			
			return Jv
		end


		function NLPModels.jtprod!(ncl::NCLModel, X::Vector{<:Float64}, v::Vector{<:Float64}, Jtv::Vector{<:Float64}) ::Vector{<:Float64}
			increment!(ncl, :neval_jtprod)
			# Test feasability
				if length(v) != ncl.meta.ncon
					error("wrong length of argument v passed to jtprod(ncl::NCLModel, X::Vector{<:Float64}, v::Vector{<:Float64}, Jtv::Vector{<:Float64}) ::Vector{<:Float64}")
				end
			
			if ncl.res_lin_cons
				Jtv .= append!(jtprod(ncl.nlp, X[1:ncl.nvar_x], v), v)
			else
				Jtv .= append!(jtprod(ncl.nlp, X[1:ncl.nvar_x], v), v[ncl.nlp.meta.nln])
			end
			return Jtv
		end




#** External function
	"""
	###########################
	Print function for NCLModel
		# TODO
	###########################
	"""
	function print(ncl::NCLModel ; 
					print_level::Int64 = 0, 
					output_file_print::Bool = false, 
					output_file_name::String = "NCLModel_display", 
					output_file::IOStream = open("NCLModel_display", write=true)
				  ) ::Nothing
		
		if print_level >= 1 # If we are supposed to print something
			if output_file_print # if it is in an output file
				if output_file_name == "NCLModel_display" # if not specified by name, may be by IOStream, so we use this one
					file = output_file
				else # Otherwise, we open the file with the requested name
					file = open(output_file_name, write=true)
				end
			else # or we print in stdout, if not specified.
				file = stdout
			end
			
			@printf(file, "NCLModel %s, ", ncl.meta.name)
			@printf(file, "    ")
		end

	end



