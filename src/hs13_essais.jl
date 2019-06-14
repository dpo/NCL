using NLPModels
using CUTEst
using NLPModelsIpopt
using AmplNLReader
include("NCLSolve.jl")


f(x) = (x[1] - 2.) ^ 2  +  x[2] ^ 2
c(x) = [(1-x[1]) ^ 3  -  x[2]]
lcon = [0.]
ucon = [Inf]
lvar = [0., 0.]
uvar = [Inf, Inf]
x0 = [-2.,-2.]

#* hand made hs13 model
	mutable struct hs13 <: AbstractNLPModel
		meta :: NLPModelMeta
		counters :: Counters
	end
		
	function hs13()
		meta = NLPModelMeta(2 ;
							ncon=1, 
							nnzh=2, 
							nnzj=2, 
							x0=[-2.0 ; -2.0], 
							lvar=lvar,
							uvar=uvar,
							lcon=lcon, 
							ucon=ucon, 
							name="hs13")

		return hs13(meta, Counters())
	end
		
	function NLPModels.obj(nlp :: hs13, x :: AbstractVector)
		increment!(nlp, :neval_obj)
		return (x[1] - 2.) ^ 2  +  x[2] ^ 2
	end
		
	function NLPModels.grad!(nlp :: hs13, x :: AbstractVector, gx :: AbstractVector)
		increment!(nlp, :neval_grad)
		gx[1] = 2.0 * (x[1] - 2)
		gx[2] = 2.0 * x[2]
		return gx
	end
		
	function NLPModels.hess(nlp :: hs13, x :: AbstractVector; obj_weight=1.0, y=[0.])
		increment!(nlp, :neval_hess)
		return [2.0 * obj_weight - 6 * y[1] * (1-x[1]) 0.0; 0.0 2]
	end
		
	function NLPModels.hess_coord(nlp :: hs13, x :: AbstractVector; obj_weight=1.0, y=[0.])
		increment!(nlp, :neval_hess)
		return ([1,2], [1,2], [2.0 * obj_weight - 6 * y * (1-x[1]), 2.])
	end
	
	function NLPModels.hess_coord!(nlp :: hs13, x :: AbstractVector, hrows::Vector{<:Int64}, hcols::Vector{<:Int64}, hvals::Vector{<:Float64} ; obj_weight=1.0, y=[0.])
		increment!(nlp, :neval_hess)
		hvals[1] = 2.0 * obj_weight - 6 * y[1] * (1-x[1])
		hvals[2] = 2
		return (hrows, hcols, hvals)
	end

	function NLPModels.hess_structure(nlp::hs13)
		increment!(nlp, :neval_hess)
		return ([1,2], [1,2])
	end
		
	function NLPModels.hprod!(nlp :: hs13, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight=1.0, y=Float64[])
		increment!(nlp, :neval_hprod)
		Hv .= [(2.0 * obj_weight - 6 * y * (1-x[1])) * v[1] ; 2. * v[2]]
		return Hv
	end

	function NLPModels.cons!(nlp :: hs13, x :: AbstractVector)
		increment!(nlp, :neval_cons)
		cx = (1-x[1]) ^ 3  -  x[2]
		return cx
	end
		
	function NLPModels.cons!(nlp :: hs13, x :: AbstractVector, cx :: AbstractVector)
		increment!(nlp, :neval_cons)
		cx = (1-x[1]) ^ 3  -  x[2]
		return cx
	end
		
	function NLPModels.jac(nlp :: hs13, x :: AbstractVector)
		increment!(nlp, :neval_jac)
		return [-3*(1-x[1])^2  -1]
	end
		
	function NLPModels.jac_coord(nlp :: hs13, x :: AbstractVector)
		increment!(nlp, :neval_jac)
		return ([1, 1], [1, 2], [-3*(1 - x[1])^2, -1])
	end

	function NLPModels.jac_coord!(nlp :: hs13, x :: AbstractVector, rows :: AbstractVector, cols :: AbstractVector, vals :: AbstractVector)
		increment!(nlp, :neval_jac)
		vals[1] = -3*(1 - x[1])^2
		vals[2] = -1
		return (rows, cols, vals)
	end
		
	function NLPModels.jprod!(nlp :: hs13, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
		increment!(nlp, :neval_jprod)
		Jv .= [(-3*(1 - x[1])^2) * v[1] - v[2]]
		return Jv
	end
		
	function NLPModels.jtprod!(nlp :: hs13, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
		increment!(nlp, :neval_jtprod)
		Jtv .= [(-3*(1 - x[1])^2) * v[1] ; - v[1]]
		return Jtv
	end

	function NLPModels.jac_structure(nlp :: hs13)
		increment!(nlp, :neval_jac)
		return ([1, 1], [1, 2])
	end


#* Differentes resolutions
	#   hand_made_hs13 = hs13()
	#	println("First resolution hand_made_hs13")
	#		resol = NLPModelsIpopt.ipopt(hand_made_hs13, tol = 1e-6, constr_viol_tol = 0.001, compl_inf_tol = 0.001, print_level = 5)
	#	
	#		println("\n\nHand made hs13 check")
	#			@show resol.solution
	#			@show KKT_check(hand_made_hs13, resol.solution, - resol.solver_specific[:multipliers_con] , resol.solver_specific[:multipliers_U] , resol.solver_specific[:multipliers_L] , 0.001, 0.001, 0.001, 3)
	#	
	#	
	#	
	#	println("\n\nCUTEst hs13 check and resolution")
	#		resol_new = NLPModelsIpopt.ipopt(CUTEst_hs13, max_iter = 5000, tol = 1e-6, constr_viol_tol = 0.001, compl_inf_tol = 0.001, print_level = 0)
	#		@show KKT_check(CUTEst_hs13, resol_new.solution, - resol_new.solver_specific[:multipliers_con] , resol_new.solver_specific[:multipliers_U] , resol_new.solver_specific[:multipliers_L] , 0.001, 0.001, 0.001, 3)
	#	
	#	
	#	
	#	ampl_hs13 = AmplModel("../../AMPL_tests/hs13.nl")
	#	println("\n\nAMPL hs13 check and resolution")
	#		resol_ampl = NLPModelsIpopt.ipopt(ampl_hs13, max_iter = 5000, tol = 1e-6, constr_viol_tol = 0.001, compl_inf_tol = 0.001, print_level = 0)
	#		@show KKT_check(CUTEst_hs13, resol_ampl.solution, - resol_ampl.solver_specific[:multipliers_con] , resol_ampl.solver_specific[:multipliers_U] , resol_ampl.solver_specific[:multipliers_L] , 0.001, 0.001, 0.001, 3)
	#		finalize(CUTEst_hs13)












nlp = hs13()
#nlp = ADNLPModel(f, x0 ; lvar=lvar, uvar=uvar, c=c, lcon=lcon, ucon=ucon, name = "HS13")
#nlp = CUTEstModel("HS13")
#nlp = CUTEstModel("TAXR13322")

ncl = NCLModel(nlp, res_lin_cons = false)



resolution = NCLSolve(nlp ;
					  max_iter_NCL = 20,
					  max_iter_solver = 500,
					  print_level_NCL = 7,
					  print_level_solver = 5,
					  linear_residuals = true,
					  output_file_print_NCL = true,
					  output_file_print_solver = false,

			    	  warm_start_init_point = "no")



finalize(nlp)

