export NCLModel
export obj, grad, grad!, cons, cons!,
       jac_structure, jac_coord!, jac_coord,
       jac, jprod, jprod!, jtprod, jtprod!,
       hess_structure, hess_coord!, hess_coord, hess, hprod, hprod!

import NLPModels: increment!
using NLPModels
using LinearAlgebra
using SparseArrays
using Test
using Printf

#!!!!! ncl.minimze dans le gradient, à voir...

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
	Subtype of AbstractNLPModel, adapted to the NCL method.
	Keeps some informations from the original AbstractNLPModel,
	and creates a new problem, modifying
		the objective function (sort of augmented lagrangian, with residuals instead of constraints)
		the constraints (with residuals)

	Process is as follows

		(nlp) | min_{x} f(x)								         | min_{X = (x,r)} F(X) = f(x) + λ' * r + ρ * ||r||²		     	(λ and ρ are parameters)
			  | subject to lvar <= x <= uvar		becomes: (ncl)   | subject to lvar <= x <= uvar, -Inf <= r <= Inf
			  | 		   lcon <= c(x) <= ucon				         | 			lcon <= c(x) + r <= ucon
"""
mutable struct NCLModel <: AbstractNLPModel
	#* I. Information about the residuals
	nlp::AbstractNLPModel # The original problem
	nx::Int # Number of variable of the nlp problem
	nr::Int # Number of residuals for the nlp problem (in fact nr = length(nln), if there are no free/infeasible constraints)
	minimize::Bool # true if the aim of the problem is to minimize, false otherwise
	res_lin_cons::Bool # Boolean to chose if you put residuals upon linear constraints (true) or not

	#* II. Constant parameters
	meta::AbstractNLPModelMeta # Informations for this problem
	counters::Counters # Counters of calls to functions like obj, grad, cons, of the problem

	#* III. Parameters for the objective function
	y::Vector{<:Float64} # Multipliers for the nlp problem, used in the lagrangian
	ρ::Float64 # Penalization of the simili lagrangian
end

"""
######################
NCLModel documentation
	Creates the NCL problem associated to the nlp in argument: from
		(nlp) | min_{x} f(x)
		      | subject to lvar <= x <= uvar
			  | 		   lcon <= c(x) <= ucon

	we create and return :
		(ncl) | min_{X = (x,r)} F(X) = f(x) + λ' * r + ρ * ||r||²		     	(λ and ρ are parameters)
			  | subject to lvar <= x <= uvar, -Inf <= r <= Inf
			  | 			lcon <= c(x) + r <= ucon

######################
"""
function NCLModel(nlp::AbstractNLPModel;  																		# Initial model
				         	res_val_init::Float64 = 0., 														# Initial value for residuals
				         	res_lin_cons::Bool = true, 															# Choose if you want residuals for linear constraints or not
				         	ρ::Float64 = 1., 																	# Initial penalty
				         	y = res_lin_cons ? zeros(Float64, nlp.meta.ncon) : zeros(Float64, nlp.meta.nnln),	# Initial multiplier, depending on the number of residuals considered
				         ) ::NCLModel

	res_lin_cons = true
	#* I. First tests
	#* I.1 Need to create a NCLModel ?
	if (nlp.meta.ncon == 0) # No need to create an NCLModel, because it is an unconstrained problem or it doesn't have non linear constraints
		@warn("\nin NCLModel($(nlp.meta.name)): the nlp problem $(nlp.meta.name) given was unconstrained, so it was returned with 0 residuals")
	elseif ((nlp.meta.nnln == 0) & !res_lin_cons) # No need to create an NCLModel, because we don't put residuals upon linear constraints (and there are not  any non linear constraint)
		@warn("\nin NCLModel($(nlp.meta.name)): the nlp problem $(nlp.meta.name) given was linearly constrained, so it was returned with 0 residuals. \nConsider setting res_lin_cons to true if you want residuals upon linear constraints.")
	end

	#* I.2 Residuals treatment
	nr = res_lin_cons ? nlp.meta.ncon : nlp.meta.nnln

	#* II. Meta field
	nx = nlp.meta.nvar
	minimize = nlp.meta.minimize
	nvar = nx + nr
	meta = NLPModelMeta(nvar;
						lvar = vcat(nlp.meta.lvar, fill!(Vector{Float64}(undef, nr), -Inf)), # No bounds upon residuals
						uvar = vcat(nlp.meta.uvar, fill!(Vector{Float64}(undef, nr), Inf)),
						x0   = vcat(nlp.meta.x0, fill!(Vector{Float64}(undef, nr), res_val_init)),
						y0   = nlp.meta.y0,
						name = nlp.meta.name * " (NCL subproblem)",
						nnzj = nlp.meta.nnzj + nr, # we add nonzeros because of residuals
						nnzh = nlp.meta.nnzh + nr,
						ncon = nlp.meta.ncon,
						lcon = nlp.meta.lcon,
						ucon = nlp.meta.ucon,
					   )

	if nlp.meta.jinf != Int[]
		error("argument problem passed to NCLModel with constraint " * string(nlp.meta.jinf) * " infeasible")
	end
	if nlp.meta.jfree != Int[]
		error("argument problem passed to NCLModel with constraint " * string(nlp.meta.jfree) * " free")
	end

	if nlp.meta.iinf != Int[]
		error("argument problem passed to NCLModel with bound constraint " * string(nlp.meta.iinf) * " infeasible")
	end

	#* III. NCLModel created:
	return NCLModel(nlp,
					nx,
					nr,
					minimize,
					res_lin_cons,
					meta,
					Counters(),
					y,
					ρ)
end

#** II. Methods

#** II.1 Objective function
function NLPModels.obj(ncl::NCLModel, xr::AbstractVector{<:Float64})::Float64
	increment!(ncl, :neval_obj)
	x = xr[1 : ncl.nx]
	r = xr[ncl.nx + 1 : ncl.nx + ncl.nr]
	y = ncl.y[1:ncl.nr]
	obj_val = obj(ncl.nlp, x)
	obj_res = y' * r + 0.5 * ncl.ρ * dot(r, r)

	if ncl.minimize
		return obj_val + obj_res
	else # argmax f(x) = argmin -f(x)
		ncl.minimize = true
		return -obj_val + obj_res
	end
end

#** II.2 Gradient of the objective function
function NLPModels.grad(ncl::NCLModel, xr::Vector{<:Float64}) ::Vector{<:Float64}
	increment!(ncl, :neval_grad)
	x = xr[1 : ncl.nx]
	r = xr[ncl.nx + 1 : ncl.nx + ncl.nr]
	y = ncl.y[1:ncl.nr]
	gx_nlp = grad(ncl.nlp, x)
	gx = vcat(gx_nlp, ncl.ρ * r + y)
	return gx
end

function NLPModels.grad!(ncl::NCLModel, X::Vector{<:Float64}, gx::Vector{<:Float64}) ::Vector{<:Float64}
	increment!(ncl, :neval_grad)

	if length(gx) != ncl.meta.nvar
		error("wrong length of argument gx passed to grad! in NCLModel
			   gx should be of length " * string(ncl.meta.nvar) * " but length " * string(length(gx)) * "given")
	end

	# Original information
	grad!(ncl.nlp, X[1:ncl.nx], gx)

	# New information (due to residuals)
	gx[ncl.nx + 1 : ncl.nx + ncl.nr] .= ncl.ρ * X[ncl.nx + 1 : ncl.nx + ncl.nr] .+ ncl.y[1:ncl.nr]

	return gx
end

#** II.3 Hessian of the Lagrangian
function NLPModels.hess(ncl::NCLModel, X::Vector{<:Float64} ; obj_weight=1.0, y=zeros) ::SparseMatrixCSC{<:Float64, Int}
	increment!(ncl, :neval_hess)

	H = sparse(hess_coord(ncl, X ; obj_weight=obj_weight, y=y)[1], hess_coord(ncl, X ; obj_weight=obj_weight, y=y)[2], hess_coord(ncl, X ; obj_weight=obj_weight, y=y)[3])

	return H
end

function NLPModels.hess_coord(ncl::NCLModel, X::Vector{<:Float64} ; obj_weight=1.0, y=zeros) ::Tuple{Vector{Int},Vector{Int},Vector{<:Float64}}
	increment!(ncl, :neval_hess)
	# Original information
	hrows, hcols, hvals = hess_coord(ncl.nlp, X[1:ncl.nx], obj_weight=obj_weight, y=y)

	# New information (due to residuals)
	append!(hrows, ncl.nx+1:ncl.meta.nvar)
	append!(hcols, ncl.nx+1:ncl.meta.nvar)
	append!(hvals, fill!(Vector{typeof(hvals[1])}(undef, ncl.nr), ncl.ρ)) # concatenate with a vector full of ncl.ρ
	return (hrows, hcols, hvals)
end

function NLPModels.hess_coord!(ncl::NCLModel, X::Vector{<:Float64}, hrows::Vector{<:Int}, hcols::Vector{<:Int}, hvals::Vector{<:Float64} ; obj_weight=1.0, y=zeros(eltype(X[1]), ncl.meta.ncon)) ::Tuple{Vector{Int},Vector{Int},Vector{<:Float64}}
	increment!(ncl, :neval_hess)
	#Pre computation
	len_hcols = length(hcols)
	orig_len = len_hcols - ncl.nr

	# Original information
	hvals[1:orig_len] .= hess_coord!(ncl.nlp, X[1:ncl.nx], hrows[1:orig_len], hcols[1:orig_len], hvals[1:orig_len], obj_weight=obj_weight, y=y)[3]

	# New information (due to residuals)
	hvals[orig_len + 1 : len_hcols] .= fill!(Vector{typeof(hvals[1])}(undef, ncl.nr), ncl.ρ) # a vector full of ncl.ρ

	return (hrows, hcols, hvals)
end

function NLPModels.hess_structure(ncl::NCLModel) ::Tuple{Vector{Int},Vector{Int}}
	increment!(ncl, :neval_hess)
	# Original information
	hrows, hcols = hess_structure(ncl.nlp)

	# New information (due to residuals)
	append!(hrows, ncl.nx+1:ncl.meta.nvar)
	append!(hcols, ncl.nx+1:ncl.meta.nvar)
	return (hrows, hcols)
end

function NLPModels.hprod(ncl::NCLModel, X::Vector{<:Float64}, v::Vector{<:Float64} ; obj_weight=1.0, y=zeros(eltype(X[1]), ncl.meta.ncon)) ::Vector{<:Float64}
	increment!(ncl, :neval_hprod)
	# Test feasibility
	if length(v) != ncl.meta.nvar
		error("wrong length of argument v passed to hprod in NCLModel
			gx should be of length " * string(ncl.meta.nvar) * " but length " * string(length(v)) * "given")
	end

	# Original information
	Hv = hprod(ncl.nlp, X[1:ncl.nx], v[1:ncl.nx], obj_weight=obj_weight, y=y)

	# New information (due to residuals)
	append!(Hv, ncl.ρ * v[ncl.nx+1:ncl.nx+ncl.nr])

	return Hv
end

function NLPModels.hprod!(ncl::NCLModel, X::Vector{<:Float64}, v::Vector{<:Float64} , Hv::Vector{<:Float64} ; obj_weight::Float64=1.0, y=zeros(eltype(X[1]), ncl.meta.ncon)) ::Vector{<:Float64}
	increment!(ncl, :neval_hprod)
	# Test feasibility
	if length(v) != ncl.meta.nvar
		error("wrong length of argument v passed to hprod in NCLModel
			gx should be of length " * string(ncl.meta.nvar) * " but length " * string(length(v)) * "given")
	end

	# Original information
	Hv[1:ncl.nx] .= hprod!(ncl.nlp, X[1:ncl.nx], v[1:ncl.nx], Hv[1:ncl.nx], obj_weight=obj_weight, y=y)

	# New information (due to residuals)
	Hv[ncl.nx+1:ncl.nx+ncl.nr] .= ncl.ρ * v[ncl.nx+1:ncl.nx+ncl.nr]

	return Hv
end

#** II.4 Constraints
function NLPModels.cons(ncl::NCLModel, X::Vector{<:Float64}) ::Vector{<:Float64}
	increment!(ncl, :neval_cons)
	# Original information
		cx = cons(ncl.nlp, X[1:ncl.nx]) # pre computation

	# New information (due to residuals)
		if ncl.res_lin_cons
			cx .+= X[ncl.nx+1:ncl.nx+ncl.nr] # a constraint on every residual
		else
			cx[ncl.nlp.meta.nln] .+= X[ncl.nx+1:ncl.nx+ncl.nr] # residual for the i-th constraint (feasible, not free and not linear (not considered in this model))
		end

	return cx
end

function NLPModels.cons!(ncl::NCLModel, X::Vector{<:Float64}, cx::Vector{<:Float64}) ::Vector{<:Float64}
	increment!(ncl, :neval_cons)
	# Original information
	cons!(ncl.nlp, X[1:ncl.nx], cx) # pre computation

	# New information (due to residuals)
	if ncl.res_lin_cons
		cx .+= X[ncl.nx+1:ncl.nx+ncl.nr]
	else
		cx[ncl.nlp.meta.nln] .+= X[ncl.nx+1:ncl.nx+ncl.nr] # residual for the i-th constraint (feasible, not free and not linear (not considered in this model))
	end

	return cx
end

#** II.5 Jacobian of the constraints vector
function NLPModels.jac(ncl::NCLModel, X::Vector{<:Float64}) ::SparseMatrixCSC{<:Float64, Int}
	increment!(ncl, :neval_jac)

	J = sparse(jac_coord(ncl, X)[1], jac_coord(ncl, X)[2], jac_coord(ncl, X)[3])

	return J
end

function NLPModels.jac_coord(ncl::NCLModel, X::Vector{<:Float64}) ::Tuple{Vector{Int},Vector{Int},Vector{<:Float64}}
	increment!(ncl, :neval_jac)
	# Original information
	jrows, jcols, jvals = jac_coord(ncl.nlp, X[1:ncl.nx])

	# New information (due to residuals)
	if ncl.res_lin_cons
		append!(jrows, 1:ncl.meta.ncon)
	else
		append!(jrows, ncl.nlp.meta.nln)
	end
	append!(jcols, ncl.nx+1 : ncl.meta.nvar)
	append!(jvals, ones(typeof(jvals[1]), ncl.nr))
	return (jrows, jcols, jvals)
end

function NLPModels.jac_coord!(ncl::NCLModel, X::Vector{<:Float64}, jrows::Vector{<:Int}, jcols::Vector{<:Int}, jvals::Vector{<:Float64}) ::Tuple{Vector{Int},Vector{Int},Vector{<:Float64}}
	increment!(ncl, :neval_jac)

	#Pre computation
	len_jcols = length(jcols)
	orig_len = len_jcols - ncl.nr

	# Test feasability
	if length(jvals) != len_jcols
		error("wrong sizes of argument jvals passed to jac_coord!(ncl::NCLModel, X::Vector{<:Float64}, jrows::Vector{<:Int}, jcols::Vector{<:Int}, jvals::Vector{<:Float64}) ::Tuple{Vector{Int},Vector{Int},Vector{<:Float64}}")
	end

	# Original informations
	jvals[1:orig_len] .= jac_coord!(ncl.nlp, X[1:ncl.nx], jrows[1:orig_len], jcols[1:orig_len], jvals[1:orig_len])[3] # we necessarily need the place for ncl.nr ones in the value array

	# New information (due to residuals)
	jvals[orig_len + 1 : len_jcols] = ones(typeof(jvals[1]), ncl.nr) # we assume length(jrows) = length(jcols) = length(jvals)

	return (jrows, jcols, jvals)
end

function NLPModels.jac_structure(ncl::NCLModel) ::Tuple{Vector{Int},Vector{Int}}
	increment!(ncl, :neval_jac)
	# Original information
	jrows, jcols = jac_structure(ncl.nlp)

	# New information (due to residuals)
	if ncl.res_lin_cons
		append!(jrows, 1:ncl.meta.ncon)
	else
		append!(jrows, ncl.nlp.meta.nln)
	end
	append!(jcols, ncl.nx+1 : ncl.meta.nvar)
	return jrows, jcols
end

function NLPModels.jprod(ncl::NCLModel, X::Vector{<:Float64}, v::Vector{<:Float64}) ::Vector{<:Float64}
	increment!(ncl, :neval_jprod)
	# Test feasability
	if length(v) != ncl.meta.nvar
		error("wrong sizes of argument v passed to jprod(ncl::NCLModel, X::Vector{<:Float64}, v::Vector{<:Float64}, Jv::Vector{<:Float64}) ::Vector{<:Float64}")
	end

	# Original information
	Jv = jprod(ncl.nlp, X[1:ncl.nx], v[1:ncl.nx])

	# New information (due to residuals)
	Resv = zeros(typeof(Jv[1,1]), ncl.meta.ncon)
	if ncl.res_lin_cons
		Resv += v[ncl.nx+1:ncl.nx+ncl.nr]
	else
		Resv[ncl.nlp.meta.nln] += v[ncl.nx+1:ncl.nx+ncl.nr]
	end

	return Jv + Resv
end

function NLPModels.jprod!(ncl::NCLModel, X::Vector{<:Float64}, v::Vector{<:Float64}, Jv::Vector{<:Float64}) ::Vector{<:Float64}
	increment!(ncl, :neval_jprod)
	# Test feasability
	if length(v) != ncl.meta.nvar
		error("wrong sizes of argument v passed to jprod!(ncl::NCLModel, X::Vector{<:Float64}, v::Vector{<:Float64}, Jv::Vector{<:Float64}) ::Vector{<:Float64}")
	end

	# Original information
	jprod!(ncl.nlp, X[1:ncl.nx], v[1:ncl.nx], Jv)

	# New information (due to residuals)
	Resv = zeros(typeof(Jv[1,1]), ncl.meta.ncon)
	if ncl.res_lin_cons
		Resv = v[ncl.nx+1:ncl.nx+ncl.nr]
	else
		Resv[ncl.nlp.meta.nln] += v[ncl.nx+1:ncl.nx+ncl.nr]
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
	Jv = jtprod(ncl.nlp, X[1:ncl.nx], v)

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
		Jtv .= append!(jtprod(ncl.nlp, X[1:ncl.nx], v), v)
	else
		Jtv .= append!(jtprod(ncl.nlp, X[1:ncl.nx], v), v[ncl.nlp.meta.nln])
	end
	return Jtv
end






#** External function
import Base.print

"""
###########################
Print function for NCLModel
	# TODO
###########################
"""
function print(ncl::NCLModel;
			         current_X::Vector{Float64} = ncl.meta.x0,
			         current_λ::Vector{Float64} = zeros(Float64, ncl.meta.ncon),
			         current_z::Vector{Float64} = zeros(Float64, ncl.meta.nvar),
			         lag_norm::Float64 = -1.,
			         print_level::Int = 1,
			         output_file_print::Bool = false,
			         output_file_name::String = "NCLModel_display",
			         output_file::IOStream = open("NCLModel_display", write=true)
			        ) ::Nothing

	file_to_close = false
	if print_level >= 1 # If we are supposed to print something
		if output_file_print # if it is in an output file
			if output_file_name == "NCLModel_display" # if not specified by name, may be by IOStream, so we use this one
				file = output_file
			else # Otherwise, we open the file with the requested name and we will close it at the end
				file = open(output_file_name, write=true)
				file_to_close = true
			end
		else # or we print in stdout, if not specified.
			file = stdout
		end
		@printf(file, "\n  ============= %s =============\n", ncl.meta.name)
		@printf(file, "    Minimization problem, with %d constraints (%d linear, %d non linear)\n", ncl.meta.ncon, ncl.meta.nlin, ncl.meta.nnln)
		@printf(file, "                               %d variables x and\n", ncl.nx)
		@printf(file, "                               %d residuals r (considered as variables)\n", ncl.nr)

		if print_level >= 2
			@printf(file, "\n    Parameters\n")
				@printf(file, "        ρ = %7.1e\n", ncl.ρ)
				@printf(file, "        ||y|| = %7.1e\n", norm(ncl.y, Inf))

			if print_level >= 3
				@printf(file, "\n    Variables\n")
					@printf(file, "        ||x|| = %7.1e\n", norm(current_X[1:ncl.nx], Inf))
					@printf(file, "        ||r|| = %7.1e\n", norm(current_X[ncl.nx+1:ncl.nx+ncl.nr], Inf))

				if print_level >= 4
					@printf(file, "\n    Functions\n")
						@printf(file, "        F(X) = %7.1e\n", obj(ncl, current_X))
						x = current_X[1:ncl.nx]

						if lag_norm >= 0.
							@printf(file, "        ||∇Lag(x, λ)|| = %7.2e\n", lag_norm)
						else
							if ncl.meta.ncon != 0
								@printf(file, "        ||∇Lag(x, λ)|| = %7.2e\n", norm(grad(ncl.nlp, x) - jtprod(ncl.nlp, x, current_λ) - current_z[1:ncl.nx], Inf))
							else
								@printf(file, "        ||∇Lag(x, λ)|| = %7.2e\n", norm(grad(ncl.nlp, x) - current_z[1:ncl.nx], Inf))
							end
						end

					@printf(file, "\n    Details: \n")

					@printf(file, "            x: \n")
					for i in 1:ncl.nx
						@printf(file, "                       lvar[%d] = %7.1e  <=?  x[%d] = %7.1e  <=?  uvar[%d] = %7.1e \n", i, ncl.meta.lvar[i], i, current_X[i], i, ncl.meta.uvar[i])
					end

					@printf(file, "            r: \n")
					for i in ncl.nx+1:ncl.nx+ncl.nr
						@printf(file, "                       lvar[%d] = %7.1e  <=?  r[%d] = %7.1e  <=?  uvar[%d] = %7.1e \n", i, ncl.meta.lvar[i], i, current_X[i], i, ncl.meta.uvar[i])
					end

					if print_level >= 5
						cx = cons(ncl, current_X)
						@printf(file, "\n            Constraint:\n")
						for i in 1:ncl.meta.ncon
							@printf(file, "                     lcon[%d] = %7.1e  <=?  c(X)[%d] = %7.1e  <=?  ucon[%d] = %7.1e \n", i, ncl.meta.lcon[i], i, cx[i], i, ncl.meta.ucon[i])
						end
					end
				end
			end
		end

		@printf(file, "  ============= end of NCLModel print =============\n")

		if file_to_close
			close(file)
		end
	end
end





import Base.println

function println(ncl::NCLModel;
				 output_file_print::Bool = false,
				 output_file_name::String = "NCLModel_display",
				 output_file::IOStream = open("NCLModel_display", write=true),
				 kwargs...
				) ::Nothing

	file_to_close = false

	if output_file_print # if it is in an output file
		if output_file_name == "NCLModel_display" # if not specified by name, may be by IOStream, so we use this one
			file = output_file
		else # Otherwise, we open the file with the requested name and we will close it at the end
			file = open(output_file_name, write=true)
			file_to_close = true
		end
	else # or we print in stdout, if not specified.
		file = stdout
	end

	print(ncl ; output_file_print=output_file_print, output_file_name=output_file_name, output_file=output_file, kwargs...)
	print(file, "\n")

	if file_to_close
		close(file)
	end
end
