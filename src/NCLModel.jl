import NLPModels: increment!
import Base.print
import Base.println
using NLPModels
using Printf
using LinearAlgebra

export NCLModel
export obj, grad, grad!, cons, cons!,
       jac_structure!, jac_coord!, jac_coord,
       jac, jprod, jprod!, jtprod, jtprod!,
       hess_structure!, hess_coord!, hess_coord, hess, hprod, hprod!


######### TODO #########
  # TODO grad_check
######### TODO #########

#** I. Model and constructor
"""
  Subtype of AbstractNLPModel, adapted to the NCL method.
  Keeps some informations from the original AbstractNLPModel,
  and creates a new problem, modifying
    the objective function (sort of augmented lagrangian, with residuals instead of constraints)
    the constraints (with residuals)

  Process is as follows

    (nlp) | min_{x} f(x)                         | min_{X = (x,r)} F(X) = f(x) + λ' * r + ρ * ||r||²          (λ and ρ are parameters)
        | subject to lvar <= x <= uvar    becomes: (ncl)   | subject to lvar <= x <= uvar, -Inf <= r <= Inf
        |        lcon <= c(x) <= ucon                |      lcon <= c(x) + r <= ucon
"""
mutable struct NCLModel <: AbstractNLPModel
  #* I. Information about the residuals
  nlp::AbstractNLPModel # The original problem
  nx::Int64 # Number of variable of the nlp problem
  nr::Int64 # Number of residuals for the nlp problem (in fact nr = length(nln), if there are no free/infeasible constraints)
  
  #* II. Constant parameters
  meta::AbstractNLPModelMeta # Informations for this problem
  counters::Counters # Counters of calls to functions like obj, grad, cons, of the problem

  #* III. Parameters for the objective function
  y::Vector{<:AbstractFloat} # Multipliers for the nlp problem, used in the lagrangian
  ρ::Float64 # Penalization of the simili lagrangian
end

"""
######################
NCLModel documentation
  Creates the NCL problem associated to the nlp in argument: from
    (nlp) | min_{x} f(x)
          | subject to lvar <= x <= uvar
        |        lcon <= c(x) <= ucon

  we create and return :
    (ncl) | min_{X = (x,r)} F(X) = f(x) + λ' * r + ρ * ||r||²         (λ and ρ are parameters)
        | subject to lvar <= x <= uvar, -Inf <= r <= Inf
        |       lcon <= c(x) + r <= ucon

######################
"""
function NCLModel(nlp::AbstractNLPModel; # Initial model
                  res_val_init::Float64 = 0., # Initial value for residuals
                  ρ::Float64 = 1., # Initial penalization
                  y = ones(Float64, nlp.meta.ncon), # Initial multiplier, depending on the number of residuals considered
                 ) ::NCLModel     

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
                    minimize = true,
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
function NLPModels.obj(ncl::NCLModel, 
                       xr::AbstractVector{<:AbstractFloat})

  increment!(ncl, :neval_obj)
  x = xr[1 : ncl.nx]
  r = xr[ncl.nx + 1 : ncl.nx + ncl.nr]

  # Original information
  obj_val = obj(ncl.nlp, x)
  ncl.nlp.meta.minimize || (obj_val *= -1)

  # New information (due to residuals)
  obj_res = ncl.y' * r + 0.5 * ncl.ρ * dot(r, r)
  return obj_val + obj_res
end

#** II.2 Gradient of the objective function
function NLPModels.grad!(ncl::NCLModel, 
                         xr::AbstractVector{<:AbstractFloat}, 
                         gx::AbstractVector{<:AbstractFloat})

  increment!(ncl, :neval_grad)

  # Original information
  x = xr[1 : ncl.nx]
  grad!(ncl.nlp, x, gx)
  ncl.nlp.meta.minimize || (gx[1:ncl.nx] .*= -1)

  # New information (due to residuals)
  r = xr[ncl.nx + 1 : ncl.nx + ncl.nr]
  gx[ncl.nx + 1 : ncl.nx + ncl.nr] .= ncl.ρ * r .+ ncl.y

  return gx
end

#** II.3 Hessian of the Lagrangian
function NLPModels.hess_structure!(ncl::NCLModel, 
                                   hrows::AbstractVector{<:Integer}, 
                                   hcols::AbstractVector{<:Integer})
  increment!(ncl, :neval_hess)
  
  # Original information
  NLPModels.hess_structure!(ncl.nlp, hrows, hcols)

  # New information (due to residuals)
  nnzh = ncl.meta.nnzh
  orig_nnzh = ncl.nlp.meta.nnzh
  hrows[orig_nnzh + 1 : nnzh] .= ncl.nx + 1 : ncl.meta.nvar
  hcols[orig_nnzh + 1 : nnzh] .= ncl.nx + 1 : ncl.meta.nvar

  return (hrows, hcols)
end

function NLPModels.hess_coord!(ncl::NCLModel,
                               xr::AbstractVector{<:AbstractFloat},
                               hrows::AbstractVector{<:Integer},
                               hcols::AbstractVector{<:Integer},
                               hvals::Vector{<:AbstractFloat};
                               obj_weight :: Float64=1.0,
                               y :: AbstractVector=zeros(ncl.nlp.meta.ncon))
  increment!(ncl, :neval_hess)

  # Pre-access
  nnzh = ncl.meta.nnzh
  orig_nnzh = ncl.nlp.meta.nnzh

  x = xr[1 : ncl.nx]

  # Original information
  ow = ncl.nlp.meta.minimize ? obj_weight : -obj_weight
  yy = ncl.nlp.meta.minimize ? y : -y
  hess_coord!(ncl.nlp, x, hrows, hcols, hvals; obj_weight=ow, y=yy)

  # New information (due to residuals)
  hvals[orig_nnzh + 1 : nnzh] .= ncl.ρ

  return (hrows, hcols, hvals)
end

function NLPModels.hprod!(ncl::NCLModel,
                          xr::AbstractVector{<:AbstractFloat},
                          v::AbstractVector{<:AbstractFloat},
                          Hv::AbstractVector{<:AbstractFloat};
                          obj_weight :: Float64=1.0,
                          y :: AbstractVector=zeros(ncl.nlp.meta.ncon))
  increment!(ncl, :neval_hprod)

  # Pre-access
  x = xr[1 : ncl.nx]

  # Original information
  ow = ncl.nlp.meta.minimize ? obj_weight : -obj_weight
  yy = ncl.nlp.meta.minimize ? y : -y
  hprod!(ncl.nlp, x, v[1 : ncl.nx], Hv; obj_weight=ow, y=yy)

  # New information (due to residuals)
  Hv[ncl.nx + 1 : ncl.nx + ncl.nr] .= ncl.ρ * v[ncl.nx + 1 : ncl.nx + ncl.nr]

  return Hv
end

function NLPModels.cons!(ncl::NCLModel, 
                         xr::AbstractVector{<:AbstractFloat}, 
                         cx::AbstractVector{<:AbstractFloat})
  increment!(ncl, :neval_cons)

  # Pre-access
  x = xr[1 : ncl.nx]
  r = xr[ncl.nx + 1 : ncl.nx + ncl.nr]
  
  # Original information
  cons!(ncl.nlp, x, cx)

  # New information (due to residuals)
  cx .+= r

  return cx
end

function NLPModels.jac_structure!(ncl::NCLModel, 
                                  jrows::AbstractVector{<:Integer}, 
                                  jcols::AbstractVector{<:Integer})
  increment!(ncl, :neval_jac)
  orig_nnzj = ncl.nlp.meta.nnzj
  nnzj = ncl.meta.nnzj

  # Original information
  NLPModels.jac_structure!(ncl.nlp, jrows, jcols)

  # New information (due to residuals)
  jrows[orig_nnzj + 1 : nnzj] .= (1 : ncl.meta.ncon)
  jcols[orig_nnzj + 1 : nnzj] .= ncl.nx+1 : ncl.meta.nvar
  return jrows, jcols
end

function NLPModels.jac_coord!(ncl::NCLModel,
                              xr::AbstractVector{<:AbstractFloat},
                              jrows::AbstractVector{<:Integer},
                              jcols::AbstractVector{<:Integer},
                              jvals::AbstractVector{<:AbstractFloat})
  increment!(ncl, :neval_jac)
  x = xr[1 : ncl.nx]
  
  # Original information
  jac_coord!(ncl.nlp, x, jrows, jcols, jvals)

  # New information (due to residuals)
  jvals[ncl.nlp.meta.nnzj + 1 : ncl.meta.nnzj] .= 1

  return (jrows, jcols, jvals)
end

function NLPModels.jprod!(ncl::NCLModel,
                          xr::AbstractVector{<:AbstractFloat},
                          v::AbstractVector{<:AbstractFloat},
                          Jv::AbstractVector{<:AbstractFloat})
  increment!(ncl, :neval_jprod)
  x = xr[1 : ncl.nx]
  vx = v[1 : ncl.nx]

  # Original information
  jprod!(ncl.nlp, x, vx, Jv)

  # New information (due to residuals)
  vr = v[ncl.nx + 1 : ncl.nx + ncl.nr]
  Resv = zeros(eltype(Jv), ncl.meta.ncon)
  Resv .= vr
  Jv .+= Resv

  return Jv
end

function NLPModels.jtprod!(ncl::NCLModel,
                           xr::AbstractVector{<:AbstractFloat},
                           v::AbstractVector{<:AbstractFloat},
                           Jtv::AbstractVector{<:AbstractFloat})
  increment!(ncl, :neval_jtprod)
  x = xr[1 : ncl.nx]
  
  # Original information
  jtprod!(ncl.nlp, x, v, Jtv)

  # New information (due to residuals)
  Jtv[ncl.nx + 1 : ncl.meta.nvar] .= v
  
  return Jtv
end






#** III Print functions
function print(ncl::NCLModel, io::IO = stdout)
  @printf(io, "%s NLP original problem :\n", ncl.nlp.meta.name)
  print(io, ncl.nlp)
  @printf(io, "\nAdded %d residuals to the previous %d variables.", ncl.nr, ncl.nx)
  
  len_y = length(ncl.y)
  begin_y = ncl.y[1 : min(3, len_y-1)]
  end_y = ncl.y[len_y]
  @printf(io, "\nCurrent y = [")
  
  for x in begin_y
    @printf(io, "%7.1e, ", x)
  end
  
  @printf(io, "..(%7.1e elements).., %7.1e]", len_y - length(begin_y) - 1, end_y)
  @printf(io, "]\nCurrent ρ = %7.1e\n", ncl.ρ)
end

function println(ncl::NCLModel, io::IO = stdout)
  print(ncl, io)
  @printf(io, "\n")
end