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

		(nlp) | min_{x} f(x)								         | min_{xr = (x,r)} F(xr) = f(x) + λ' * r + ρ * ||r||²		     	(λ and ρ are parameters)
			  | subject to lvar <= x <= uvar		becomes: (ncl)   | subject to lvar <= x <= uvar, -Inf <= r <= Inf
			  | 		   lcon <= c(x) <= ucon				         | 			lcon <= c(x) + r <= ucon
"""
mutable struct NCLModel <: AbstractNLPModel
	#* I. Information about the residuals
	nlp::AbstractNLPModel # The original problem
	nx::Integer # Number of variable of the nlp problem
	nr::Integer # Number of residuals for the nlp problem (in fact nr = length(nln), if there are no free/infeasible constraints)
	
	#* II. Constant parameters
	meta::AbstractNLPModelMeta # Informations for this problem
	counters::Counters # Counters of calls to functions like obj, grad, cons, of the problem

	#* III. Parameters for the objective function
	y::AbstractVector{<:AbstractFloat} # Multipliers for the nlp problem, used in the lagrangian
	ρ::AbstractFloat # Penalization of the simili lagrangian
end

"""
######################
NCLModel documentation
	Creates the NCL problem associated to the nlp in argument: from
		(nlp) | min_{x} f(x)
		      | subject to lvar <= x <= uvar
			  | 		   lcon <= c(x) <= ucon

	we create and return :
		(ncl) | min_{xr = (x,r)} F(xr) = f(x) + λ' * r + ρ * ||r||²		     	(λ and ρ are parameters)
			  | subject to lvar <= x <= uvar, -Inf <= r <= Inf
			  | 			lcon <= c(x) + r <= ucon

######################
"""
function NCLModel(nlp::AbstractNLPModel;  																		# Initial model

				         	res_val_init::AbstractFloat = 0., 														# Initial value for residuals
				         	ρ::AbstractFloat = 1., 																	# Initial penalty
				         	y = zeros(AbstractFloat, nlp.meta.ncon)	# Initial multiplier, depending on the number of residuals considered
				         ) #::NCLModel

	#* I. First tests
	#* I.1 Need to create a NCLModel ?
	if (nlp.meta.ncon == 0) # No need to create an NCLModel, because it is an unconstrained problem or it doesn't have non linear constraints
		@debug("\nin NCLModel($(nlp.meta.name)): the nlp problem $(nlp.meta.name) given was unconstrained, so it was returned with 0 residuals")
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

	if nlp.meta.jinf != Integer[]
		error("argument problem passed to NCLModel with constraint " * string(nlp.meta.jinf) * " infeasible")
	end
	if nlp.meta.jfree != Integer[]
		error("argument problem passed to NCLModel with constraint " * string(nlp.meta.jfree) * " free")
	end

	if nlp.meta.iinf != Integer[]
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

#to return both objective functions :
function objnlp_objncl(ncl::NCLModel, xr::AbstractVector{<:Float64})#::Float64
	increment!(ncl, :neval_obj)
	x = xr[1 : ncl.nx]
	r = xr[ncl.nx + 1 : ncl.nx + ncl.nr]
	y = ncl.y[1:ncl.nr]

	#Original information
	obj_nlp = obj(ncl.nlp, x)
	real_obj_nlp = obj_nlp
	if !(ncl.nlp.meta.minimize) 
		obj_nlp *= -1
	end

	# New information (due to residuals)
	obj_res = y' * r + 0.5 * ncl.ρ * dot(r, r)

	return real_obj_nlp, obj_nlp + obj_res
end

#** II.2 Gradient of the objective function
function NLPModels.grad!(ncl::NCLModel, xr::Vector{<:Float64}, gx::Vector{<:Float64}) #::Vector{<:Float64}
	increment!(ncl, :neval_grad)

	if length(gx) != ncl.meta.nvar
		error("wrong length of argument gx passed to grad! in NCLModel
			   gx should be of length " * string(ncl.meta.nvar) * " but length " * string(length(gx)) * "given")
	end

	# Original information
	grad!(ncl.nlp, xr[1:ncl.nx], gx)
	if !(ncl.nlp.meta.minimize)
		gx *= -1
	end
	
	# New information (due to residuals)
	gx[ncl.nx + 1 : ncl.nx + ncl.nr] .= ncl.ρ * xr[ncl.nx + 1 : ncl.nx + ncl.nr] .+ ncl.y[1:ncl.nr]

	return gx
end

#** II.3 Hessian of the Lagrangian
function NLPModels.hess(ncl::NCLModel, xr::Vector{<:Float64} ; obj_weight=1.0, y=zeros) #::SparseMatrixCSC{<:Float64, <:Integer}
	increment!(ncl, :neval_hess)

	H = sparse(hess_coord(ncl, xr ; obj_weight=obj_weight, y=y)...)

	return H
end

function NLPModels.hess_coord!(ncl::NCLModel, xr::Vector{<:Float64}, hrows::Vector{<:Integer}, hcols::Vector{<:Integer}, hvals::Vector{<:Float64} ; obj_weight=1.0, y=zeros(eltype(xr[1]), ncl.meta.ncon)) #::Tuple{Vector{<:Integer},Vector{<:Integer},Vector{<:Float64}}
	increment!(ncl, :neval_hess)
	#Pre computation
	len_hcols = length(hcols)
	orig_len = len_hcols - ncl.nr

	# Original information
	hess_coord!(ncl.nlp, xr[1:ncl.nx], hrows, hcols, hvals, obj_weight=obj_weight, y=y)
	if !(ncl.nlp.meta.minimize) 
		hvals *= -1
	end
	
	# New information (due to residuals)
	hvals[orig_len + 1 : len_hcols] .= ncl.ρ # a vector full of ncl.ρ

	return (hrows, hcols, hvals)
end

function NLPModels.hess_structure!(ncl::NCLModel, hrows::Vector{<:Integer}, hcols::Vector{<:Integer}) #::Tuple{Vector{<:Integer},Vector{<:Integer}}
	increment!(ncl, :neval_hess)
	rows = fill!(Vector{Int64}(undef, length(hrows)), 0)
	cols = fill!(Vector{Int64}(undef, length(hcols)), 0)

	# Original information
	NLPModels.hess_structure!(ncl.nlp, rows, cols)
	
	hrows .= rows
	hcols .= cols

	# New information (due to residuals)
	hrows[end - (ncl.meta.nvar - ncl.nx)+1: end] .= [i for i in ncl.nx+1:ncl.meta.nvar]
	hcols[end - (ncl.meta.nvar - ncl.nx)+1: end] .= [i for i in ncl.nx+1:ncl.meta.nvar]
	return (hrows, hcols)
end

function NLPModels.hprod!(ncl::NCLModel, xr::Vector{<:Float64}, v::Vector{<:Float64} , Hv::Vector{<:Float64} ; obj_weight::Float64=1.0, y=zeros(eltype(xr[1]), ncl.meta.ncon)) #::Vector{<:Float64}
	increment!(ncl, :neval_hprod)
	# Test feasibility
	if length(v) != ncl.meta.nvar
		error("wrong length of argument v passed to hprod in NCLModel
			gx should be of length " * string(ncl.meta.nvar) * " but length " * string(length(v)) * "given")
	end

	# Original information
	Hv[1:ncl.nx] .= hprod!(ncl.nlp, xr[1:ncl.nx], v[1:ncl.nx], Hv[1:ncl.nx], obj_weight=obj_weight, y=y)
	if !(ncl.nlp.meta.minimize) 
		Hv *= -1
	end

	# New information (due to residuals)
	Hv[ncl.nx+1:ncl.nx+ncl.nr] .= ncl.ρ * v[ncl.nx+1:ncl.nx+ncl.nr]

	return Hv
end

#** II.4 Constraints
function NLPModels.cons!(ncl::NCLModel, xr::Vector{<:Float64}, cx::Vector{<:Float64}) #::Vector{<:Float64}
	increment!(ncl, :neval_cons)
	# Original information
	cons!(ncl.nlp, xr[1:ncl.nx], cx) # pre computation

	# New information (due to residuals)
	cx .+= xr[ncl.nx+1:ncl.nx+ncl.nr] # residual for the i-th constraint
	
	return cx
end

#** II.5 Jacobian of the constraints vector
function NLPModels.jac(ncl::NCLModel, xr::Vector{<:Float64}) ::SparseMatrixCSC{<:Float64, <:Integer}
	increment!(ncl, :neval_jac)
	J = sparse(jac_coord(ncl, xr)...)
	return J
end

function NLPModels.jac_coord!(ncl::NCLModel, xr::Vector{<:Float64}, jrows::Vector{<:Integer}, jcols::Vector{<:Integer}, jvals::Vector{<:Float64}) #::Tuple{Vector{<:Integer},Vector{<:Integer},Vector{<:Float64}}
	increment!(ncl, :neval_jac)

	#Pre computation
	len_jcols = length(jcols)
	orig_len = len_jcols - ncl.nr

	# Test feasability
	if length(jvals) != len_jcols
		error("wrong sizes of argument jvals passed to jac_coord!(ncl::NCLModel, xr::Vector{<:Float64}, jrows::Vector{<:Integer}, jcols::Vector{<:Integer}, jvals::Vector{<:Float64}) ::Tuple{Vector{<:Integer},Vector{<:Integer},Vector{<:Float64}}")
	end

	# Original informations
	jac_coord!(ncl.nlp, xr[1:ncl.nx], jrows, jcols, jvals) # we necessarily need the place for ncl.nr ones in the value array

	# New information (due to residuals)
	jvals[orig_len + 1 : len_jcols] = ones(typeof(jvals[1]), ncl.nr) # we assume length(jrows) = length(jcols) = length(jvals)

	return (jrows, jcols, jvals)
end

function NLPModels.jac_structure!(ncl::NCLModel, jrows::Vector{<:Integer}, jcols::Vector{<:Integer}) #::Tuple{Vector{<:Integer},Vector{<:Integer}}
	increment!(ncl, :neval_jac)
	# Original information
	rows = fill!(Vector{Int64}(undef, length(jrows)), 0)
	cols = fill!(Vector{Int64}(undef, length(jcols)), 0)

	NLPModels.jac_structure!(ncl.nlp, rows, cols)
	
	jrows .= rows
	jcols .= cols

	# New information (due to residuals)
	jrows[end - (ncl.nlp.meta.ncon)+1: end] .= [i for i in 1:ncl.nlp.meta.ncon]
	jcols[end - (ncl.meta.nvar - ncl.nx)+1: end] .= [i for i in ncl.nx+1:ncl.meta.nvar]

	return jrows, jcols
end

function NLPModels.jprod!(ncl::NCLModel, xr::Vector{<:Float64}, v::Vector{<:Float64}, Jv::Vector{<:Float64}) #::Vector{<:Float64}
	increment!(ncl, :neval_jprod)
	# Test feasability
	if length(v) != ncl.meta.nvar
		error("wrong sizes of argument v passed to jprod!(ncl::NCLModel, xr::AbstractVector{<:AbstractFloat}, v::AbstractVector{<:AbstractFloat}, Jv::AbstractVector{<:AbstractFloat}) ::AbstractVector{<:AbstractFloat}")
	end

	# Original information
	jprod!(ncl.nlp, xr[1:ncl.nx], v[1:ncl.nx], Jv)

	# New information (due to residuals)
	Resv = zeros(typeof(Jv[1,1]), ncl.meta.ncon)
	Resv = v[ncl.nx+1:ncl.nx+ncl.nr]
	Jv .+= Resv

	return Jv
end

function NLPModels.jtprod!(ncl::NCLModel, xr::Vector{<:Float64}, v::Vector{<:Float64}, Jtv::Vector{<:Float64}) #::Vector{<:Float64}
	increment!(ncl, :neval_jtprod)
	# Test feasability
	if length(v) != ncl.meta.ncon
		error("wrong length of argument v passed to jtprod(ncl::NCLModel, xr::AbstractVector{<:AbstractFloat}, v::AbstractVector{<:AbstractFloat}, Jtv::AbstractVector{<:AbstractFloat}) ::AbstractVector{<:AbstractFloat}")
	end

	Jtv .= append!(jtprod(ncl.nlp, xr[1:ncl.nx], v), v)

	return Jtv
end


#** III Print functions
function print(io::IO, ncl::NCLModel)
	print(io, "$(ncl.nlp.meta.name) NLP original problem :\n")
	print(io, ncl.nlp)
	print(io, "\nNCLModel new information")
	print(io, "\nnr = $(ncl.nr) residuals")
	
	if ncl.nr > 0
	  if ncl.nr < 5
		print(io, "\nCurrent y = ", ncl.y)
	  else
		len_y = length(ncl.y)
		begin_y = ncl.y[1 : min(3, len_y-1)]
		end_y = ncl.y[len_y]
		print(io, "\nCurrent y = [")
		
		for x in begin_y
		  print(io, "$x, ")
		end
		
		print(io, "...($(len_y - length(begin_y)-1) elements)..., $end_y]")
	  end
	else
	  print(io, "\nncl.y = ∅")
	end
  
	print(io, "\nCurrent ρ = $(ncl.ρ)\n")
end
  
  # Dispatch and related print functions
function println(io::IO, ncl::NCLModel)
	print(io, ncl)
	print(io, "\n")
end
  
function print(ncl::NCLModel)
	print(stdout, ncl)
end
  
function println(ncl::NCLModel)
	println(stdout, ncl)
end