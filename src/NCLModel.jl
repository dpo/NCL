export NCLModel
export obj, grad, grad!, cons, cons!,
       jac_structure!, jac_coord!, jac_coord,
       jac, jprod, jprod!, jtprod, jtprod!,
       hess_structure!, hess_coord!, hess_coord, hess, hprod, hprod!

import NLPModels: increment!
import Base.print
import Base.println

#!!!!! ncl.minimize dans le gradient, à voir...

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
				         	ρ::Float64 = 1., 																	# Initial penalty
				         	y = zeros(Float64, nlp.meta.ncon)	# Initial multiplier, depending on the number of residuals considered
				         ) #::NCLModel

	#* I. First tests
	#* I.1 Need to create a NCLModel ?
	if (nlp.meta.ncon == 0) # No need to create an NCLModel, because it is an unconstrained problem or it doesn't have non linear constraints
		@warn("\nin NCLModel($(nlp.meta.name)): the nlp problem $(nlp.meta.name) given was unconstrained, so it was returned with 0 residuals")
	end

	#* I.2 Residuals treatment
	nr = nlp.meta.ncon

	#* II. Meta field
	nx = nlp.meta.nvar
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
					meta,
					Counters(),
					y,
					ρ)
end

#** II. Methods

#** II.1 Objective function
function NLPModels.obj(ncl::NCLModel, xr::AbstractVector{<:Float64})#::Float64
	increment!(ncl, :neval_obj)
	x = xr[1 : ncl.nx]
	r = xr[ncl.nx + 1 : ncl.nx + ncl.nr]
	y = ncl.y[1:ncl.nr]

	#Original information
	obj_nlp = obj(ncl.nlp, x)
	if !(ncl.nlp.meta.minimize) 
		obj_nlp *= -1
	end

	# New information (due to residuals)
	obj_res = y' * r + 0.5 * ncl.ρ * dot(r, r)

	return obj_nlp + obj_res
end

#** II.2 Gradient of the objective function
function NLPModels.grad!(ncl::NCLModel, X::Vector{<:Float64}, gx::Vector{<:Float64}) #::Vector{<:Float64}
	increment!(ncl, :neval_grad)

	if length(gx) != ncl.meta.nvar
		error("wrong length of argument gx passed to grad! in NCLModel
			   gx should be of length " * string(ncl.meta.nvar) * " but length " * string(length(gx)) * "given")
	end

	# Original information
	grad!(ncl.nlp, X[1:ncl.nx], gx)
	if !(ncl.nlp.meta.minimize)
		gx *= -1
	end
	
	# New information (due to residuals)
	gx[ncl.nx + 1 : ncl.nx + ncl.nr] .= ncl.ρ * X[ncl.nx + 1 : ncl.nx + ncl.nr] .+ ncl.y[1:ncl.nr]

	return gx
end

#** II.3 Hessian of the Lagrangian
function NLPModels.hess(ncl::NCLModel, X::Vector{<:Float64} ; obj_weight=1.0, y=zeros) #::SparseMatrixCSC{<:Float64, Int}
	increment!(ncl, :neval_hess)

	H = sparse(hess_coord(ncl, X ; obj_weight=obj_weight, y=y)...)

	return H
end

function NLPModels.hess_coord!(ncl::NCLModel, X::Vector{<:Float64}, hrows::Vector{<:Int}, hcols::Vector{<:Int}, hvals::Vector{<:Float64} ; obj_weight=1.0, y=zeros(eltype(X[1]), ncl.meta.ncon)) #::Tuple{Vector{Int},Vector{Int},Vector{<:Float64}}
	increment!(ncl, :neval_hess)
	#Pre computation
	len_hcols = length(hcols)
	orig_len = len_hcols - ncl.nr

	# Original information
	hvals[1:orig_len] .= hess_coord!(ncl.nlp, X[1:ncl.nx], hrows[1:orig_len], hcols[1:orig_len], hvals[1:orig_len], obj_weight=obj_weight, y=y)[3]
	if !(ncl.nlp.meta.minimize) 
		hvals *= -1
	end
	
	# New information (due to residuals)
	hvals[orig_len + 1 : len_hcols] .= ncl.ρ # a vector full of ncl.ρ

	return (hrows, hcols, hvals)
end

function NLPModels.hess_structure!(ncl::NCLModel, hrows::Vector{<:Int}, hcols::Vector{<:Int}) #::Tuple{Vector{Int},Vector{Int}}
	increment!(ncl, :neval_hess)

	# Original information
	NLPModels.hess_structure!(ncl.nlp, hrows, hcols)

	# New information (due to residuals)
	hrows[end - (ncl.meta.nvar - ncl.nx)+1: end] .= [i for i in ncl.nx+1:ncl.meta.nvar]
	hcols[end - (ncl.meta.nvar - ncl.nx)+1: end] .= [i for i in ncl.nx+1:ncl.meta.nvar]
	return (hrows, hcols)
end

function NLPModels.hprod!(ncl::NCLModel, X::Vector{<:Float64}, v::Vector{<:Float64} , Hv::Vector{<:Float64} ; obj_weight::Float64=1.0, y=zeros(eltype(X[1]), ncl.meta.ncon)) #::Vector{<:Float64}
	increment!(ncl, :neval_hprod)
	# Test feasibility
	if length(v) != ncl.meta.nvar
		error("wrong length of argument v passed to hprod in NCLModel
			gx should be of length " * string(ncl.meta.nvar) * " but length " * string(length(v)) * "given")
	end

	# Original information
	Hv[1:ncl.nx] .= hprod!(ncl.nlp, X[1:ncl.nx], v[1:ncl.nx], Hv[1:ncl.nx], obj_weight=obj_weight, y=y)
	if !(ncl.nlp.meta.minimize) 
		Hv *= -1
	end

	# New information (due to residuals)
	Hv[ncl.nx+1:ncl.nx+ncl.nr] .= ncl.ρ * v[ncl.nx+1:ncl.nx+ncl.nr]

	return Hv
end

#** II.4 Constraints
function NLPModels.cons!(ncl::NCLModel, X::Vector{<:Float64}, cx::Vector{<:Float64}) #::Vector{<:Float64}
	increment!(ncl, :neval_cons)
	# Original information
	cons!(ncl.nlp, X[1:ncl.nx], cx) # pre computation

	# New information (due to residuals)
	cx .+= X[ncl.nx+1:ncl.nx+ncl.nr] # residual for the i-th constraint
	

	return cx
end

#** II.5 Jacobian of the constraints vector
function NLPModels.jac(ncl::NCLModel, X::Vector{<:Float64}) ::SparseMatrixCSC{<:Float64, Int}
	increment!(ncl, :neval_jac)
	J = sparse(jac_coord(ncl, X)...)
	return J
end

function NLPModels.jac_coord!(ncl::NCLModel, X::Vector{<:Float64}, jrows::Vector{<:Int}, jcols::Vector{<:Int}, jvals::Vector{<:Float64}) #::Tuple{Vector{Int},Vector{Int},Vector{<:Float64}}
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

function NLPModels.jac_structure!(ncl::NCLModel, jrows::Vector{Int}, jcols::Vector{Int}) #::Tuple{Vector{Int},Vector{Int}}
	increment!(ncl, :neval_jac)
	# Original information
	NLPModels.jac_structure!(ncl.nlp, jrows, jcols)

	# New information (due to residuals)
	jrows[end - (ncl.nlp.meta.ncon)+1: end] .= [i for i in 1:ncl.nlp.meta.ncon]
	jcols[end - (ncl.meta.nvar - ncl.nx)+1: end] .= [i for i in ncl.nx+1:ncl.meta.nvar]

	return jrows, jcols
end

function NLPModels.jprod!(ncl::NCLModel, X::Vector{<:Float64}, v::Vector{<:Float64}, Jv::Vector{<:Float64}) #::Vector{<:Float64}
	increment!(ncl, :neval_jprod)
	# Test feasability
	if length(v) != ncl.meta.nvar
		error("wrong sizes of argument v passed to jprod!(ncl::NCLModel, X::Vector{<:Float64}, v::Vector{<:Float64}, Jv::Vector{<:Float64}) ::Vector{<:Float64}")
	end

	# Original information
	jprod!(ncl.nlp, X[1:ncl.nx], v[1:ncl.nx], Jv)

	# New information (due to residuals)
	Resv = zeros(typeof(Jv[1,1]), ncl.meta.ncon)
	Resv = v[ncl.nx+1:ncl.nx+ncl.nr]
	Jv .+= Resv

	return Jv
end

function NLPModels.jtprod!(ncl::NCLModel, X::Vector{<:Float64}, v::Vector{<:Float64}, Jtv::Vector{<:Float64}) #::Vector{<:Float64}
	increment!(ncl, :neval_jtprod)
	# Test feasability
	if length(v) != ncl.meta.ncon
		error("wrong length of argument v passed to jtprod(ncl::NCLModel, X::Vector{<:Float64}, v::Vector{<:Float64}, Jtv::Vector{<:Float64}) ::Vector{<:Float64}")
	end

	Jtv .= append!(jtprod(ncl.nlp, X[1:ncl.nx], v), v)

	return Jtv
end


#** III Print functions
function print(ncl::NCLModel, io::IO = stdout)
	@printf(io, "$(ncl.nlp.meta.name) NLP original problem :\n")
	@printf(io, ncl.nlp)
	@printf(io, "\nAdded $(ncl.nr) residuals to the previous $(ncl.nx) variables.")
	@printf(io, "\nCurrent y = [$(ncl.y[1 : min(3, length(ncl.y)-1)])...$(ncl.y[length(ncl.y)])]")
	@printf(io, "\nCurrent ρ = $(ncl.ρ)\n")
end

function println(ncl::NCLModel, io::IO = stdout)
	print(ncl, io)
	@printf(io, "\n")
end