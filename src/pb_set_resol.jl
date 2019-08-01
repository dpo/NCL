export pb_set_resolution_data

"""
Function solving sets of problems (`CUTEst`, `NLPModels` or `AmplModel`).
"""
function pb_set_resolution_data(; #No arguments, only key-word arguments
																#* Optimization parameters
																tol::Float64 = 1e-6,
																constr_viol_tol::Float64 = 1e-6,
																compl_inf_tol::Float64 = 1e-4,
																acc_factor::Float64 = 100.,
																max_iter_NCL = 30,

																#* CUTEst arguments
																cutest_pb_set::Vector{String} = String[],
																cutest_pb_index::Vector{Int} = [i for i in 1:length(cutest_pb_set)],

																#* NLP Arguments
																nlp_pb_set::Vector{<:AbstractNLPModel} = AbstractNLPModel[],
																nlp_pb_index::Vector{Int} = [i for i in 1:length(nlp_pb_set)],

																#* AmplNLReader Arguments
																ampl_pb_set::Vector{String} = String[],
																ampl_pb_index::Vector{Int} = [i for i in 1:length(ampl_pb_set)],
																ampl_pb_dir_path::String = "./",

																#* Solver arguments
																solver::Vector{String} = ["ipopt", "nclres", "nclkkt"], #can contain ipopt
																																	# knitro (not yet, to be updated later...)
																																	# nclres (stops when norm(r) is small enough, not checking kkt conditions during iterations)
																																	# nclkkt (stops when fitting KKT conditions, or fitting to acceptable level)
												
																print_level_iter::Int = 0,
																print_level_checks::Int = 0,
																print_level_NCL_solver::Int = 0,
																max_iter_solver::Int = 1000,

																#* Files
																create_profile::Bool = true,
																create_latex_table::Bool = true,
																profile_name = "default_profile_name",
																latex_table_name = "default_latex_table_name.tex"
																)#::Nothing

	n_solver = length(solver)
	n_cutest = length(cutest_pb_index)
	n_nlp = length(nlp_pb_index)
	n_ampl = length(ampl_pb_index)


	info_cutest::Array{Int, 2} = Array{Int, 2}(undef, n_cutest, 2) # 1: nvar, 2: ncon
	names_cutest::Vector{String} = Vector{String}(undef, n_cutest)
	time_cutest::Array{Real, 3} = Array{Real, 3}(undef, n_solver, n_cutest, 5) # (n_solver rows, n_cutest cols, 2 in depth) (pb, solver, 1): neval_obj, (pb, solver, 2): neval_con
	resol_cutest::Array{Dict{Symbol,Any}, 2} = Array{Dict{Symbol,Any}, 2}(undef, n_solver, n_cutest) # contains : iter, obj_val, mult_norm, r_norm, internal_msg
	kkt_cutest::Array{Dict{Symbol,Any}, 2} = Array{Dict{Symbol,Any}, 2}(undef, n_solver, n_cutest)

	info_nlp::Array{Int, 2} = Array{Int, 2}(undef, n_nlp, 2) # 1: nvar, 2: ncon
	names_nlp::Vector{String} = Vector{String}(undef, n_nlp)
	time_nlp::Array{Real, 3} = Array{Real, 3}(undef, n_solver, n_nlp, 5) # (n_solver rows, n_nlp cols, 2 in depth) (pb, solver, 1): neval_obj, (pb, solver, 2): neval_con
	resol_nlp::Array{Dict{Symbol,Any}, 2} = Array{Dict{Symbol,Any}, 2}(undef, n_solver, n_nlp)
	kkt_nlp::Array{Dict{Symbol,Any}, 2} = Array{Dict{Symbol,Any}, 2}(undef, n_solver, n_nlp)

	info_ampl::Array{Int, 2} = Array{Int, 2}(undef, n_ampl, 2) # 1: nvar, 2: ncon
	names_ampl::Vector{String} = Vector{String}(undef, n_ampl)
	time_ampl::Array{Real, 3} = Array{Real, 3}(undef, n_solver, n_ampl, 5) # (n_solver rows, n_nlp cols, 2 in depth) (pb, solver, 1): neval_obj, (pb, solver, 2): neval_con
	resol_ampl::Array{Dict{Symbol,Any}, 2} = Array{Dict{Symbol,Any}, 2}(undef, n_solver, n_ampl)
	kkt_ampl::Array{Dict{Symbol,Any}, 2} = Array{Dict{Symbol,Any}, 2}(undef, n_solver, n_ampl)


	#** I. CUTEst problem set
	k = 0
	for i in cutest_pb_index
		k += 1
		#** I.1 Problem
		nlp = CUTEstModel(cutest_pb_set[i])

		info_cutest[k, 1] = nlp.meta.nvar
		info_cutest[k, 2] = nlp.meta.ncon

		names_cutest[k] = nlp.meta.name

		#** I.2 Resolution
		for i in 1:n_solver
			if solver[i] == "nclres"
				reset!(nlp.counters)
				resol_nclres, time_cutest[i, k, 3], time_cutest[i, k, 4], time_cutest[i, k, 5], memallocs = @timed NCLSolve(nlp ;
																																																										#
																																																										tol = tol,
																																																										constr_viol_tol = constr_viol_tol,
																																																										compl_inf_tol = compl_inf_tol,
																																																										acc_factor = acc_factor,
																																																										print_level_NCL = print_level_iter,
																																																										#
																																																										max_iter_NCL = max_iter_NCL,
																																																										KKT_checking = false)

				time_cutest[i, k, 1] = nlp.counters.neval_obj
				time_cutest[i, k, 2] = nlp.counters.neval_cons

				kkt_cutest[i, k] = KKTCheck(nlp,
																		resol_nclres.solution,
																		resol_nclres.solver_specific[:multipliers_con],
																		resol_nclres.solver_specific[:multipliers_U],
																		resol_nclres.solver_specific[:multipliers_L] ;
																		print_level = print_level_checks,
																		tol = tol,
																		constr_viol_tol = constr_viol_tol,
																		compl_inf_tol = compl_inf_tol,
																		)

				resol_cutest[i, k] = Dict(:iter => resol_nclres.iter,
																	:obj_val => resol_nclres.objective,
																	:mult_norm => norm(vcat(resol_nclres.solver_specific[:multipliers_con], (resol_nclres.solver_specific[:multipliers_L] - resol_nclres.solver_specific[:multipliers_U])), Inf),
																	:r_norm => haskey(resol_nclres.solver_specific, :residuals) ? norm(resol_nclres.solver_specific[:residuals], Inf) : 0.,
																	:internal_msg => Symbol(resol_nclres.solver_specific[:internal_msg])
																	)
			end

			if solver[i] == "nclkkt"
				reset!(nlp.counters)
				resol_nclkkt, time_cutest[i, k, 3], time_cutest[i, k, 4], time_cutest[i, k, 5], memallocs = @timed NCLSolve(nlp ;
																																																										max_iter_NCL = max_iter_NCL,
																																																										print_level_NCL = print_level_iter,
																																																										tol = tol,
																																																										constr_viol_tol = constr_viol_tol,
																																																										compl_inf_tol = compl_inf_tol,
																																																										acc_factor = acc_factor,
																																																										max_iter_solver = max_iter_solver,
																																																										KKT_checking = true)

				time_cutest[i, k, 1] = nlp.counters.neval_obj
				time_cutest[i, k, 2] = nlp.counters.neval_cons

				kkt_cutest[i, k] = KKTCheck(nlp,
																		resol_nclkkt.solution,
																		resol_nclkkt.solver_specific[:multipliers_con],
																		resol_nclkkt.solver_specific[:multipliers_U],
																		resol_nclkkt.solver_specific[:multipliers_L] ;
																		print_level = print_level_checks,
																		tol = tol,
																		constr_viol_tol = constr_viol_tol,
																		compl_inf_tol = compl_inf_tol,
																		)

				resol_cutest[i, k] = Dict(:iter => resol_nclkkt.iter,
																	:obj_val => resol_nclkkt.objective,
																	:mult_norm => norm(vcat(resol_nclkkt.solver_specific[:multipliers_con], (resol_nclkkt.solver_specific[:multipliers_L] - resol_nclkkt.solver_specific[:multipliers_U])), Inf),
																	:r_norm => haskey(resol_nclkkt.solver_specific, :residuals) ? norm(resol_nclkkt.solver_specific[:residuals], Inf) : 0.,
																	:internal_msg => resol_nclkkt.solver_specific[:internal_msg]
																	)
			end

			if solver[i] == "ipopt"
				reset!(nlp.counters)
				resol_solver, time_cutest[i, k, 3], time_cutest[i, k, 4], time_cutest[i, k, 5], memallocs = @timed NLPModelsIpopt.ipopt(nlp ; max_iter = max_iter_solver,
																																																																tol = tol,
																																																																constr_viol_tol = constr_viol_tol,
																																																																compl_inf_tol = compl_inf_tol,
																																																																print_level = 0,
																																																																)


				time_cutest[i, k, 1] = nlp.counters.neval_obj
				time_cutest[i, k, 2] = nlp.counters.neval_cons
        
				kkt_cutest[i, k] = KKTCheck(nlp,
																		resol_solver.solution,
																		resol_solver.solver_specific[:multipliers_con],
																		resol_solver.solver_specific[:multipliers_U],
																		resol_solver.solver_specific[:multipliers_L] ;
																		print_level = print_level_checks,
																		tol = tol,
																		constr_viol_tol = constr_viol_tol,
																		compl_inf_tol = compl_inf_tol)

				resol_cutest[i, k] = Dict(:iter => resol_solver.iter,
																	:obj_val => resol_solver.objective,
																	:mult_norm => norm(vcat(resol_solver.solver_specific[:multipliers_con], (resol_solver.solver_specific[:multipliers_L] - resol_solver.solver_specific[:multipliers_U])), Inf),
																	:r_norm => haskey(resol_solver.solver_specific, :residuals) ? norm(resol_solver.solver_specific[:residuals], Inf) : 0.,
																	:internal_msg => resol_solver.solver_specific[:internal_msg]
																	)
			end
		end
		finalize(nlp)
	end


	#** II. NLP problem set
	k = 0
	for i in nlp_pb_index
		k += 1
		#** II.1 Problem
		nlp = nlp_pb_set[i]

		info_nlp[k, 1] = nlp.meta.nvar
		info_nlp[k, 2] = nlp.meta.ncon

		names_nlp[k] = nlp.meta.name

		#** II.2 Resolution
		for i in 1:n_solver
			if solver[i] == "nclres"
				reset!(nlp.counters)
				resol_nclres, time_nlp[i, k, 3], time_nlp[i, k, 4], time_nlp[i, k, 5], memallocs = @timed NCLSolve(nlp ;
																																																						max_iter_NCL = max_iter_NCL,
																																																						print_level_NCL = print_level_iter,
																																																						tol = tol,
																																																						constr_viol_tol = constr_viol_tol,
																																																						compl_inf_tol = compl_inf_tol,
																																																						acc_factor = acc_factor,
																																																						max_iter_solver = max_iter_solver,
																																																						KKT_checking = false)

				time_nlp[i, k, 1] = nlp.counters.neval_obj
				time_nlp[i, k, 2] = nlp.counters.neval_cons

				kkt_nlp[i, k] = KKTCheck(nlp,
																	resol_nclres.solution,
																	resol_nclres.solver_specific[:multipliers_con],
																	resol_nclres.solver_specific[:multipliers_U],
																	resol_nclres.solver_specific[:multipliers_L] ;
																	print_level = print_level_checks,
																	tol = tol,
																	constr_viol_tol = constr_viol_tol,
																	compl_inf_tol = compl_inf_tol)

				resol_nlp[i, k] = Dict(:iter => resol_nclres.iter,
															:obj_val => resol_nclres.objective,
															:mult_norm => norm(vcat(resol_nclres.solver_specific[:multipliers_con], (resol_nclres.solver_specific[:multipliers_L] - resol_nclres.solver_specific[:multipliers_U])), Inf),
															:r_norm => haskey(resol_nclres.solver_specific, :residuals) ? norm(resol_nclres.solver_specific[:residuals], Inf) : 0.,
															:internal_msg => resol_nclres.solver_specific[:internal_msg]
															)
			end

			if solver[i] == "nclkkt"
				reset!(nlp.counters)
				resol_nclkkt, time_nlp[i, k, 3], time_nlp[i, k, 4], time_nlp[i, k, 5], memallocs = @timed NCLSolve(nlp ;
																																																						max_iter_NCL = max_iter_NCL,
																																																						print_level_NCL = print_level_iter,
																																																						tol = tol,
																																																						constr_viol_tol = constr_viol_tol,
																																																						compl_inf_tol = compl_inf_tol,
																																																						acc_factor = acc_factor,
																																																						max_iter_solver = max_iter_solver,
																																																						KKT_checking = true)

				time_nlp[i, k, 1] = nlp.counters.neval_obj
				time_nlp[i, k, 2] = nlp.counters.neval_cons

				kkt_nlp[i, k] = KKTCheck(nlp,
																resol_nclkkt.solution,
																resol_nclkkt.solver_specific[:multipliers_con],
																resol_nclkkt.solver_specific[:multipliers_U],
																resol_nclkkt.solver_specific[:multipliers_L] ;
																print_level = print_level_checks,
																tol = tol,
																constr_viol_tol = constr_viol_tol,
																compl_inf_tol = compl_inf_tol)

				resol_nlp[i, k] = Dict(:iter => resol_nclkkt.iter,
															:obj_val => resol_nclkkt.objective,
															:mult_norm => norm(vcat(resol_nclkkt.solver_specific[:multipliers_con], (resol_nclkkt.solver_specific[:multipliers_L] - resol_nclkkt.solver_specific[:multipliers_U])), Inf),
															:r_norm => haskey(resol_nclkkt.solver_specific, :residuals) ? norm(resol_nclkkt.solver_specific[:residuals], Inf) : 0.,
															:internal_msg => resol_nclkkt.solver_specific[:internal_msg]
															)
			end

			if solver[i] == "ipopt"
				reset!(nlp.counters)
				resol_solver, time_nlp[i, k, 3], time_nlp[i, k, 4], time_nlp[i, k, 5], memallocs = @timed NLPModelsIpopt.ipopt(nlp ; 
																																																												max_iter = max_iter_solver,
																																																												tol = tol,
																																																												print_level = 0,#print_level_iter,
																																																												constr_viol_tol = constr_viol_tol,
																																																												compl_inf_tol = compl_inf_tol,
																																																												)
																															   
				time_nlp[i, k, 1] = nlp.counters.neval_obj
				time_nlp[i, k, 2] = nlp.counters.neval_cons

				kkt_nlp[i, k] = KKTCheck(nlp,
																resol_solver.solution,
																resol_solver.solver_specific[:multipliers_con],
																resol_solver.solver_specific[:multipliers_U],
																resol_solver.solver_specific[:multipliers_L] ;
																print_level = print_level_checks,
																tol = tol,
																constr_viol_tol = constr_viol_tol,
																compl_inf_tol = compl_inf_tol)

				resol_nlp[i, k] = Dict(:iter => resol_solver.iter,
															:obj_val => resol_solver.objective,
															:mult_norm => norm(vcat(resol_solver.solver_specific[:multipliers_con], (resol_solver.solver_specific[:multipliers_L] - resol_solver.solver_specific[:multipliers_U])), Inf),
															:r_norm => haskey(resol_solver.solver_specific, :residuals) ? norm(resol_solver.solver_specific[:residuals], Inf) : 0.,
															:internal_msg => resol_solver.solver_specific[:internal_msg]
															)
			end
		end

	end


	#** III. AMPL problem set
	k = 0
	actual_dir = pwd()
	cd(ampl_pb_dir_path)
	for i in ampl_pb_index
		k += 1
    
		#** III.1 Problem
		tax_name = ampl_pb_set[i]
		
		if !isfile(tax_name * ".nl")
			run(Cmd(["ampl", "-og" * tax_name, tax_name * ".mod", tax_name * ".dat"]))
		end
    
		ampl_model = AmplModel(tax_name * ".nl")
    
		info_ampl[k, 1] = ampl_model.meta.nvar
		info_ampl[k, 2] = ampl_model.meta.ncon

		names_ampl[k] = ampl_model.meta.name

		#** III.2 Resolution
		for i in 1:n_solver
			if solver[i] == "nclres"
				reset!(ampl_model.counters)
				resol_nclres, time_ampl[i, k, 3], time_ampl[i, k, 4], time_ampl[i, k, 5], memallocs = @timed NCLSolve(ampl_model ;
																																																							print_level_NCL = print_level_iter,
																																																							print_level_solver = print_level_NCL_solver,
																																																							max_iter_NCL = max_iter_NCL,
																																																							tol = tol,
																																																							constr_viol_tol = constr_viol_tol,
																																																							compl_inf_tol = compl_inf_tol,
																																																							acc_factor = acc_factor,
																																																							max_iter_solver = max_iter_solver,
																																																							KKT_checking = false)

				time_ampl[i, k, 1] = ampl_model.counters.neval_obj
				time_ampl[i, k, 2] = ampl_model.counters.neval_cons

				kkt_ampl[i, k] = KKTCheck(ampl_model,
																	resol_nclres.solution,
																	resol_nclres.solver_specific[:multipliers_con],
																	resol_nclres.solver_specific[:multipliers_U],
																	resol_nclres.solver_specific[:multipliers_L] ;
																	print_level = print_level_checks,
																	tol = tol,
																	constr_viol_tol = constr_viol_tol,
																	compl_inf_tol = compl_inf_tol)

				resol_ampl[i, k] = Dict(:iter => resol_nclres.iter,
																:obj_val => resol_nclres.objective,
																:mult_norm => norm(vcat(resol_nclres.solver_specific[:multipliers_con], (resol_nclres.solver_specific[:multipliers_L] - resol_nclres.solver_specific[:multipliers_U])), Inf),
																:r_norm => haskey(resol_nclres.solver_specific, :residuals) ? norm(resol_nclres.solver_specific[:residuals], Inf) : 0.,
																:internal_msg => resol_nclres.solver_specific[:internal_msg]
																)
			end

			if solver[i] == "nclkkt"
				reset!(ampl_model.counters)
				resol_nclkkt, time_ampl[i, k, 3], time_ampl[i, k, 4], time_ampl[i, k, 5], memallocs = @timed NCLSolve(ampl_model ;
																																																							max_iter_NCL = max_iter_NCL,
																																																							print_level_NCL = print_level_iter,
																																																							tol = tol,
																																																							constr_viol_tol = constr_viol_tol,
																																																							compl_inf_tol = compl_inf_tol,
																																																							acc_factor = acc_factor,
																																																							max_iter_solver = max_iter_solver,
																																																							KKT_checking = true)

				time_ampl[i, k, 1] = ampl_model.counters.neval_obj
				time_ampl[i, k, 2] = ampl_model.counters.neval_cons

				kkt_ampl[i, k] = KKTCheck(ampl_model,
																	resol_nclkkt.solution,
																	resol_nclkkt.solver_specific[:multipliers_con],
																	resol_nclkkt.solver_specific[:multipliers_U],
																	resol_nclkkt.solver_specific[:multipliers_L] ;
																	print_level = print_level_checks,
																	tol = tol,
																	constr_viol_tol = constr_viol_tol,
																	compl_inf_tol = compl_inf_tol)

				resol_ampl[i, k] = Dict(:iter => resol_nclkkt.iter,
																:obj_val => resol_nclkkt.objective,
																:mult_norm => norm(vcat(resol_nclkkt.solver_specific[:multipliers_con], (resol_nclkkt.solver_specific[:multipliers_L] - resol_nclkkt.solver_specific[:multipliers_U])), Inf),
																:r_norm => haskey(resol_nclkkt.solver_specific, :residuals) ? norm(resol_nclkkt.solver_specific[:residuals], Inf) : 0.,
																:internal_msg => resol_nclkkt.solver_specific[:internal_msg]
																)
			end

			if solver[i] == "ipopt"
				reset!(ampl_model.counters)
				resol_solver, time_ampl[i, k, 3], time_ampl[i, k, 4], time_ampl[i, k, 5], memallocs = @timed NLPModelsIpopt.ipopt(ampl_model ;
																																																													max_iter = max_iter_solver,
																																																													tol = tol,
																																																													constr_viol_tol = constr_viol_tol,
																																																													compl_inf_tol = compl_inf_tol,
																																																													print_level = 0#print_level_iter,
																																																													)

				time_ampl[i, k, 1] = ampl_model.counters.neval_obj
				time_ampl[i, k, 2] = ampl_model.counters.neval_cons

				kkt_ampl[i, k] = KKTCheck(ampl_model,
																	resol_solver.solution,
																	resol_solver.solver_specific[:multipliers_con],
																	resol_solver.solver_specific[:multipliers_U],
																	resol_solver.solver_specific[:multipliers_L] ;
																	print_level = print_level_checks,
																	tol = tol,
																	constr_viol_tol = constr_viol_tol,
																	compl_inf_tol = compl_inf_tol)

				resol_ampl[i, k] = Dict(:iter => resol_solver.iter,
																:obj_val => resol_solver.objective,
																:mult_norm => norm(vcat(resol_solver.solver_specific[:multipliers_con], (resol_solver.solver_specific[:multipliers_L] - resol_solver.solver_specific[:multipliers_U])), Inf),
																:r_norm => haskey(resol_solver.solver_specific, :residuals) ? norm(resol_solver.solver_specific[:residuals], Inf) : 0.,
																:internal_msg => resol_solver.solver_specific[:internal_msg]
																)
			end
		end

		finalize(ampl_model)
	end
	(length(ampl_pb_index) >= 1) & (ampl_pb_dir_path != "./") && cd(actual_dir) # goes back to actual directory if needed


	#** IV. Data frames
	info = vcat(info_cutest, info_nlp, info_ampl)
	names = vcat(names_cutest, names_nlp, names_ampl)

	resol = hcat(resol_cutest, resol_nlp, resol_ampl)
	kkt = hcat(kkt_cutest, kkt_nlp, kkt_ampl)
	time = hcat(time_cutest, time_nlp, time_ampl)

	n_pb = n_cutest + n_nlp + n_ampl


	if create_latex_table
		stats = Dict(Symbol(solver[i]) => DataFrame(:problem	=> [names[k] for k in 1:n_pb],
																								:id			=> [k for k in 1:n_pb],
																								:nvar 		=> [info[k, 1] for k in 1:n_pb],
																								:ncon		=> [info[k, 2] for k in 1:n_pb],

																								:niter 		=> [resolution[:iter] for resolution in resol[i, :]],
																								:f 			=> [resolution[:obj_val] for resolution in resol[i, :]],

																								:feval 		=> [time[i, k, 1] for k in 1:n_pb],
																								:ceval 		=> [time[i, k, 2] for k in 1:n_pb],
																								:time		=> [time[i, k, 3] for k in 1:n_pb],
																								:bytes		=> [time[i, k, 4] for k in 1:n_pb],
																								:gctime		=> [time[i, k, 5] for k in 1:n_pb],

																								:feas 		=> [kkt_res[:primal_feas] for kkt_res in kkt[i, :]],
																								:compl 		=> [kkt_res[:complementarity_feas] for kkt_res in kkt[i, :]],
																								:mult_norm 	=> [resolution[:mult_norm] for resolution in resol[i, :]],
																								:lag_norm 	=> [kkt_res[:dual_feas] for kkt_res in kkt[i, :]],
																								:r_norm 	=> [resolution[:r_norm] for resolution in resol[i, :]],

																								:solve_succeeded => [resolution[:internal_msg] for resolution in resol[i, :]],
																								:r_opti 	=> [Symbol(resolution[:r_norm] <= tol) for resolution in resol[i, :]],
																								:r_acc_opti	=> [Symbol(resolution[:r_norm] <= acc_factor * tol) for resolution in resol[i, :]],
																								:kkt_opti 	=> [Symbol(kkt_res[:optimal]) for kkt_res in kkt[i, :]],
																								:kkt_acc_opti => [Symbol(kkt_res[:acceptable]) for kkt_res in kkt[i, :]])
					for i in 1:n_solver)
	
	
		hdr_override = Dict(:problem => "\\textbf{Problem}",
												:nvar => "\$n_{var}\$",
												:ncon => "\$n_{con}\$",
												:niter => "\$n_{iter}\$",
												:f => "\$f\\left(x \\right)\$",
												:feval => "\$f_{eval}\$",
												:ceval => "\$c_{eval}\$",
												:time => "\$time\$",
												:bytes => "\$bytes\$",
												:gctime => "\$gctime\$",
												:feas => "\$feas\$",
												:compl => "\$compl\$",
												:mult_norm => "\$ \\left\\Vert \\lambda \\right\\Vert\$",
												:lag_norm => "\$ \\left\\Vert \\nabla_{x} L \\right\\Vert\$",
												:r_norm => "\$ \\left\\Vert r \\right\\Vert\$",
												:solve_succeeded => "Succeeded ?",
												:r_opti => "\$ \\left\\Vert r \\right\\Vert \\,\\leq ?\\, \\eta_\\infty\$",
												:r_acc_opti => "\$ \\left\\Vert r \\right\\Vert \\,\\leq ?\\, \\eta_{acc}\$",
												:kkt_opti => "\$KKT\$",
												:kkt_acc_opti => "\$KKT_{acc}\$"
											)

		N = [:niter, :f, :feval, :ceval, :time, :bytes, :gctime, :feas, :compl, :mult_norm, :lag_norm, :r_norm, :solve_succeeded, :r_opti, :r_acc_opti, :kkt_opti, :kkt_acc_opti]
		df_res = join(stats, N ; invariant_cols = [:problem, :nvar, :ncon], hdr_override = hdr_override)


		#* V. Results
		if !isdir("./res/")
			mkdir("./res/")
		end

		ltx_file = open("./res/$latex_table_name", write = true)
		latex_table(ltx_file, df_res)
		close(ltx_file)
	end

	if create_profile
		solved(df) = (df.solve_succeeded .== :Solve_Succeeded)
		acc_solved(df) = ((df.solve_succeeded .== :Solve_Succeeded) .| (df.solve_succeeded .== :Solved_To_Acceptable_Level))
		kkt_opti(df) = (df.kkt_opti .== Symbol(true))
		kkt_acc_opti(df) = (df.kkt_acc_opti .== Symbol(true))
		compare = [df -> .!solved(df)*Inf + df.feval + df.ceval, 
							 df -> .!solved(df)*Inf + .!acc_solved(df)*Inf + df.feval + df.ceval, 
							 df -> .!kkt_opti(df) * Inf + df.feval + df.ceval, 
							 df -> .!kkt_opti(df) * Inf + .!kkt_acc_opti(df) * Inf + df.feval + df.ceval]

		comparison_names = ["optimal + f_eval + c_eval", "acceptable + f_eval + c_eval", "KKToptimal + f_eval + c_eval", "KKTacceptable + f_eval + c_eval"]
		p = profile_solvers(stats, compare, comparison_names)

		Plots.svg(p, "./res/$profile_name")
	end

	return nothing
end
