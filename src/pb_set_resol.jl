export pb_set_resolution_data

function pb_set_resolution_files( ; #No arguments, only key-word arguments
						#* Common arguments
							path_res_folder::String = "/home/perselie/Bureau/projet/ncl/res/",

						#* Optimization parameters
							tol::Float64 = 1e-8,
							constr_viol_tol::Float64 = 1e-6,
							compl_inf_tol::Float64 = 1e-4,
							KKT_checking::Bool = false,
							linear_residuals = true,

						#* CUTEst arguments
							cutest_generic_pb_name::String = "CUTEst_HS",
							cutest_pb_set::Vector{String} = String[],
							cutest_pb_index_set::Vector{Int} = Int[],

						#* Solver arguments
							solver::String = "ipopt",
							print_level_solver::Int = 0,
							max_iter_solver::Int = 1000,

						#* NLP Arguments
							nlp_generic_pb_name::String = "NLP_HS",
							nlp_pb_set::Vector{<:AbstractNLPModel} = AbstractNLPModel[],
							nlp_pb_index_set::Vector{Int} = Int[],

						#* Latex ?
							generate_latex::Bool = true
						)



	#** I. CUTEst problem set
		#** I.0 Directory check
			if !isdir(path_res_folder * cutest_generic_pb_name * "/")
				mkdir(path_res_folder * cutest_generic_pb_name * "/")
			end

			file_cutest = open(path_res_folder * cutest_generic_pb_name * "/" * cutest_generic_pb_name * ".log", write=true)

		for i in cutest_pb_index_set
			#** I.1 Problem
				nlp = CUTEstModel(cutest_pb_set[i])

			#** I.2 Resolution
				#* I.2.1 NCL resolution
					reset!(nlp.counters)
					resol = NCLSolve(nlp ;
							max_iter_NCL = 20,
							tol = tol,
							constr_viol_tol = constr_viol_tol,
							compl_inf_tol = compl_inf_tol,
							max_iter_solver = 1000,
							print_level_NCL = 6,
							print_level_solver = 0,
							linear_residuals = linear_residuals,
							KKT_checking = KKT_checking,
							output_file_print_NCL = true,
							output_file_print_solver = false,
							output_file_NCL = file_cutest,
							warm_start = true)

					@printf(file_cutest, "\n=================\n")

					D = KKTCheck(nlp,
								resol.solution,
								resol.solver_specific[:multipliers_con],
								resol.solver_specific[:multipliers_U],
								resol.solver_specific[:multipliers_L] ;
								tol = tol,
								constr_viol_tol = constr_viol_tol,
								compl_inf_tol = compl_inf_tol,
								print_level = 3,
								output_file_print = true,
								output_file = file_cutest
								)

				#* I.2.1 solver resolution

					#if solver == "ipopt"
					reset!(nlp.counters)
					resol_solver = NLPModelsIpopt.ipopt(nlp ; max_iter = max_iter_solver,
													tol = tol,
													constr_viol_tol = constr_viol_tol,
													compl_inf_tol = compl_inf_tol,
													print_level = print_level_solver)


					@printf(file_cutest, "\n\n=== Checking solver resolution ===\n")

					D_solver = KKTCheck(nlp,
										resol.solution,
										resol.solver_specific[:multipliers_con],
										resol.solver_specific[:multipliers_U],
										resol.solver_specific[:multipliers_L] ;

										tol = tol,
										constr_viol_tol = constr_viol_tol,
										compl_inf_tol = compl_inf_tol,

										print_level = 3,
										output_file_print = true,
										output_file = file_cutest
										)


			#** I.3 Print summary
				summary_path = path_res_folder * cutest_generic_pb_name * "/summary_" * cutest_generic_pb_name * "_" * nlp.meta.name * ".txt"
				file_summary = open(summary_path, write=true)

				#* I.3.1 NCL summary
					@printf(file_summary, "name = \"%s\" \n", nlp.meta.name)
					@printf(file_summary, "nvar = %d\n", nlp.meta.nvar)
					@printf(file_summary, "ncon = %d\n", nlp.meta.ncon)
					@printf(file_summary, "iter = %d\n", resol.iter)
					@printf(file_summary, "obj_val = %7.2e\n", resol.objective)

					@printf(file_summary, "obj_call = %d\n", resol.counters.counters.neval_obj)
					@printf(file_summary, "cons_call = %d\n", resol.counters.counters.neval_cons)

					@printf(file_summary, "norm_mult = %7.2e\n", norm(vcat(resol.solver_specific[:multipliers_con], (resol.solver_specific[:multipliers_L] - resol.solver_specific[:multipliers_U])), Inf))
					@printf(file_summary, "norm_grad_lag = %9.2e\n", D["norm_grad_lag"])
					@printf(file_summary, "norm_r = %9.2e\n", haskey(resol.solver_specific, :residuals) ? norm(resol.solver_specific[:residuals]) : 0.)

					@printf(file_summary, "optimal_res = %s\n", haskey(resol.solver_specific, :residuals) ? (norm(resol.solver_specific[:residuals]) <= constr_viol_tol) : true)
					@printf(file_summary, "optimal_kkt = %s\n", string(D["optimal"]))
					@printf(file_summary, "acceptable_kkt = %s\n", string(D["acceptable"]))


				#* I.3.2 Solver summary
					@printf(file_summary, "obj_val_solver = %7.2e\n", resol_solver.objective)
					@printf(file_summary, "obj_call_solver = %d\n", nlp.counters.neval_obj)
					@printf(file_summary, "cons_call_solver = %d\n", resol.counters.counters.neval_cons)

					@printf(file_summary, "norm_mult_solver = %7.2e\n", norm(vcat(resol_solver.solver_specific[:multipliers_con], (resol_solver.solver_specific[:multipliers_L] - resol_solver.solver_specific[:multipliers_U])), Inf))
					@printf(file_summary, "norm_grad_lag_solver = %7.2e\n", D_solver["norm_grad_lag"])

					@printf(file_summary, "optimal_solver = %s\n", string(resol_solver.solver_specific[:internal_msg] == Symbol("Solve_Succeeded"))),
					@printf(file_summary, "optimal_kkt_solver = %s\n", string(D_solver["optimal"]))
					@printf(file_summary, "acceptable_kkt_solver = %s\n", string(D_solver["acceptable"]))


				close(file_summary)


			@printf(file_cutest, "\n============= End of resolution =============\n\n\n\n\n")
			finalize(nlp)
		end

		close(file_cutest)



	#** II. NLP problem set
		#** I.0 Directory check
			if !isdir(path_res_folder * nlp_generic_pb_name * "/")
				mkdir(path_res_folder * nlp_generic_pb_name * "/")
			end

			file_nlp = open(path_res_folder *  nlp_generic_pb_name * "/" * nlp_generic_pb_name * ".log", write=true)

		for i in nlp_pb_index_set
			#** II.1 Problem
				nlp = nlp_pb_set[i]

			#** II.2 Resolution
				#** II.2.1 NCL resolution
					reset!(nlp.counters)
					resol = NCLSolve(nlp ;
							max_iter_NCL = 20,
							max_iter_solver = 1000,
							tol = tol,
							constr_viol_tol = constr_viol_tol,
							compl_inf_tol = compl_inf_tol,
							print_level_NCL = 6,
							print_level_solver = 0,
							linear_residuals = linear_residuals,
							KKT_checking = KKT_checking,
							output_file_print_NCL = true,
							output_file_print_solver = false,
							output_file_NCL = file_nlp,
							warm_start = true)

					D = KKTCheck(nlp,
							resol.solution,
							resol.solver_specific[:multipliers_con],
							resol.solver_specific[:multipliers_U],
							resol.solver_specific[:multipliers_L] ;
							tol = tol,
							constr_viol_tol = constr_viol_tol,
							compl_inf_tol = compl_inf_tol,

							print_level = 3,
							output_file_print = true,
							output_file = file_nlp
						   )


				#** II.2.2 Solver resolution
					#if solver == "ipopt"
					reset!(nlp.counters)

					resol_solver = NLPModelsIpopt.ipopt(nlp ; max_iter = max_iter_solver,
													tol = tol,
													constr_viol_tol = constr_viol_tol,
													compl_inf_tol = compl_inf_tol,
													print_level = print_level_solver)



					@printf(file_nlp, "\n\n=== Checking solver resolution ===\n")

					D_solver = KKTCheck(nlp,
										resol.solution,
										resol.solver_specific[:multipliers_con],
										resol.solver_specific[:multipliers_U],
										resol.solver_specific[:multipliers_L] ;

										tol = tol,
										constr_viol_tol = constr_viol_tol,
										compl_inf_tol = compl_inf_tol,

										print_level = 3,
										output_file_print = true,
										output_file = file_nlp
										)

			#** II.3 Print summary
				summary_path = path_res_folder * "$nlp_generic_pb_name/summary_" * nlp_generic_pb_name * "_" * nlp.meta.name * ".txt"
				file_summary = open(summary_path, write=true)

				#** II.3.1 NCL summary
					@printf(file_summary, "name = \"%s\" \n", nlp.meta.name)
					@printf(file_summary, "nvar = %d\n", nlp.meta.nvar)
					@printf(file_summary, "ncon = %d\n", nlp.meta.ncon)
					@printf(file_summary, "iter = %d\n", resol.iter)
					@printf(file_summary, "obj_val = %7.2e\n", resol.objective)

					@printf(file_summary, "obj_call = %d\n", resol.counters.counters.neval_obj)
					@printf(file_summary, "cons_call = %d\n", resol.counters.counters.neval_cons)

					@printf(file_summary, "norm_mult = %7.2e\n", norm(vcat(resol.solver_specific[:multipliers_con], (resol.solver_specific[:multipliers_L] - resol.solver_specific[:multipliers_U])), Inf))
					@printf(file_summary, "norm_grad_lag = %7.2e\n", D["norm_grad_lag"])

					@printf(file_summary, "obj_val = %7.2e\n", resol.objective)

					@printf(file_summary, "norm_r = %7.2e\n", haskey(resol.solver_specific, :residuals) ? norm(resol.solver_specific[:residuals]) : 0.)
					@printf(file_summary, "optimal_res = %s\n", haskey(resol.solver_specific, :residuals) ? (norm(resol.solver_specific[:residuals]) <= constr_viol_tol) : true)
					@printf(file_summary, "optimal_kkt = %s\n", string(D["optimal"]))
					@printf(file_summary, "acceptable_kkt = %s\n", string(D["acceptable"]))


				#** II.3.2 Solver summary
					@printf(file_summary, "obj_val_solver = %7.2e\n", resol_solver.objective)
					@printf(file_summary, "obj_call_solver = %d\n", resol_solver.counters.counters.neval_obj)
					@printf(file_summary, "cons_call_solver = %d\n", resol_solver.counters.neval_cons)

					@printf(file_summary, "norm_mult_solver = %7.2e\n", norm(vcat(resol_solver.solver_specific[:multipliers_con], (resol_solver.solver_specific[:multipliers_L] - resol_solver.solver_specific[:multipliers_U])), Inf))
					@printf(file_summary, "norm_grad_lag_solver = %7.2e\n", D_solver["norm_grad_lag"])

					@printf(file_summary, "optimal_solver = %s\n", string(resol_solver.solver_specific[:internal_msg] == Symbol("Solve_Succeeded"))),
					@printf(file_summary, "optimal_kkt_solver = %s\n", string(D_solver["optimal"]))
					@printf(file_summary, "acceptable_kkt_solver = %s\n", string(D_solver["acceptable"]))

				close(file_summary)


			@printf(file_nlp, "\n============= End of resolution =============\n\n\n\n\n")

		end

		close(file_nlp)

	if generate_latex
		res_tabular("./res/latex.tex")
	end
end

"""
#################
res_tabular function
	Arguments
	- outputFile: path of the output file

	Prerequisites:
	- Each subfolder must contain text files
	- Each text file correspond to the resolution of one problem
	- Each text file contains a variable  #TODO


    This function creates a LaTeX file, with a table, showing some details of the resolution of problems in pb_set, in argument.
    At the beginning of each table, you have : The name of the problem considered
                                               The number of variables,
                                                          of constraints (linear and non linear),
                                                          of iterations of NCL until termination,
                                               the final objective value.
                                               the final residual norm
                                               the lagrangian gradient norm
                                               the optimality check with residual norm
                                               the optimality check with KKT conditions
#################
"""
function res_tabular(outputFile::String ;
                      resultFolder::String = "/home/perselie/Bureau/projet/ncl/res/"
                    )

    if !isdir(resultFolder)
        mkpath(resultFolder)
    end

    # Open the latex output file
    fout = open(outputFile, "w")

    # Print the latex file output
    println(fout, raw"""\documentclass{article}

    \usepackage[french]{babel}
    \usepackage [utf8] {inputenc} % utf-8 / latin1
    \usepackage[T1]{fontenc}
	\usepackage{multicol}
	\usepackage[dvipsnames]{xcolor}
	\usepackage{lscape}
    \usepackage{tikz}
    \def\checkmark{\tikz\fill[scale=0.4](0,.35) -- (.25,0) -- (1,.7) -- (.25,.15) -- cycle;}

    \setlength{\hoffset}{-18pt}
    \setlength{\oddsidemargin}{0pt} % Marge gauche sur pages impaires
    \setlength{\evensidemargin}{9pt} % Marge gauche sur pages paires
    \setlength{\marginparwidth}{54pt} % Largeur de note dans la marge
    \setlength{\textwidth}{481pt} % Largeur de la zone de texte (17cm)
    \setlength{\voffset}{-18pt} % Bon pour DOS
    \setlength{\marginparsep}{7pt} % Séparation de la marge
    \setlength{\topmargin}{0pt} % Pas de marge en haut
    \setlength{\headheight}{13pt} % Haut de page
    \setlength{\headsep}{10pt} % Entre le haut de page et le texte
    \setlength{\footskip}{27pt} % Bas de page + séparation
    \setlength{\textheight}{668pt} % Hauteur de la zone de texte (25cm)

	\begin{document}
	\begin{landscape}""")

		header = raw"""
    \begin{center}
	\renewcommand{\arraystretch}{1.4}
    \begin{tabular}{c"""

    # List of all the instances solved by at least one resolution method
    solvedProblems = Array{String, 1}()

    # For each file in the result folder
    for file in readdir(resultFolder)

        path = resultFolder * file

        # If it is a subfolder
        if isdir(path)
            # Add all its files in the solvedProblems array
            for subfile in filter(x->occursin(".txt", x), readdir(path))
                solvedProblems = vcat(solvedProblems, "$path/$subfile")
            end
        elseif occursin(".txt", path)
            solvedProblems = vcat(solvedProblems, path)
        end
    end

    # Only keep one string for each instance solved
    unique(solvedProblems)

	header_solver = header

    header *= "ccccccccccccccc}\n\t\\hline\n" #seven columns

    # column names
	header *= "\\\\\n
			   \\textbf{Problem}
			   & \$n_{var}\$
			   & \$n_{con}\$
			   & \$n_{iter}\$

			   & \$f\\left(x\\right)\$

			   & \$n_{eval}\\left(f\\right)\$

			   & \$\\left\\Vert y \\right\\Vert_\\infty\$
			   & \$\\left\\Vert \\nabla_x L \\right\\Vert_\\infty\$

			   & \$\\left\\Vert r \\right\\Vert_\\infty\$
			   & KKT



			   & \$f_s\\left(x\\right)\$

			   & \$n_{eval}\\left(f_s\\right)\$

			   & \$\\left\\Vert y \\right\\Vert_\\infty\$
			   & \$\\left\\Vert \\nabla_x L_s \\right\\Vert_\\infty\$

			   & \\textbf{Solved}
			   & \$\\text{\\textbf{KKT}}_{\\text{\\textbf{s}}}\$

			   "

    header *= "\\\\\\hline\n"

    footer = raw"""\hline\end{tabular}\end{center}"""

    println(fout, header)

    # On each page an array will contain at most maxInstancePerPage lines with results
    maxInstancePerPage = 27
    id = 1

    # For each solved problems
    for solvedProblem in solvedProblems

        # If we do not start a new array on a new page
        if rem(id, maxInstancePerPage) == 0
            println(fout, footer, "\\newpage")
            println(fout, header)
        end

        # Replace the potential underscores '_' in file names
        #print(fout, replace(solvedProblem, "_" => "\\_"))
        replace(solvedProblem, "_" => "\\_")

        #path = resultFolder * "/" * solvedProblem

        include(solvedProblem)



		#! Finish !



		# NCL summary
		println(fout, name * " & ",
					  nvar, " & ",
					  ncon, " & ",
					  iter, " & ",

					  obj_val, " & ",
					  obj_call, " & ",

					  norm_mult, " & ",

					  norm_grad_lag, " & ",

					  optimal_res ? "\\color{green}" : "\\color{red}",
					  norm_r, " & ",

					  optimal_kkt ? "\\color{green} \\checkmark"
					  			  : acceptable_kkt ? "\\color{orange} \\checkmark"
					  				 			    : "\\color{red} \$\\times\$", " & ")

		# Solver summary
		println(fout, obj_val_solver, " & ",

					  obj_call_solver, " & ",

					  norm_mult_solver, " & ",

					  norm_grad_lag_solver, " & ",
					  optimal_solver ? "\\color{green} \\checkmark"
								     : "\\color{red} \$\\times\$", " & ",
					  optimal_kkt_solver ? "\\color{green} \\checkmark"
								         : acceptable_kkt_solver ? "\\color{orange} \\checkmark"
										 						  : "\\color{red} \$\\times\$")


        println(fout, "\\\\")

        id += 1
    end

    # Print the end of the latex file
    println(fout, footer)

    println(fout, "\\end{landscape}\\end{document}")

    close(fout)
end
#################



function pb_set_resolution_data(; #No arguments, only key-word arguments
						#* Optimization parameters
						tol::Float64 = 1e-6,
						constr_viol_tol::Float64 = 1e-6,
						compl_inf_tol::Float64 = 1e-4,
						acc_factor::Float64 = 100.,
						linear_residuals = true,
						max_iter_NCL = 30,

						#* CUTEst arguments
						cutest_pb_set::Vector{String} = String[],
						cutest_pb_index_set::Vector{Int} = [i for i in 1:length(cutest_pb_set)],

						#* NLP Arguments
						nlp_pb_set::Vector{<:AbstractNLPModel} = AbstractNLPModel[],
						nlp_pb_index_set::Vector{Int} = [i for i in 1:length(nlp_pb_set)],

						#* AmplNLReader Arguments
						ampl_pb_set::Vector{String} = String[],
						ampl_pb_index_set::Vector{Int} = [i for i in 1:length(ampl_pb_set)],

						#* Solver arguments
						solver::Vector{String} = ["ipopt", "nclres", "nclkkt"], #can contain ipopt
																						   # knitro
																						   # nclres (stops when norm(r) is small enough, not checking kkt conditions during iterations)
																						   # nclkkt (stops when fitting KKT conditions, or fitting to acceptable level)
    
						print_level_iter::Int = 0,
						print_level_checks::Int = 0,
						print_level_NCL_solver::Int = 0,
						max_iter_solver::Int = 1000
						)::Nothing

	n_solver = length(solver)
	n_cutest = length(cutest_pb_index_set)
	n_nlp = length(nlp_pb_index_set)
	n_ampl = length(ampl_pb_index_set)


	info_cutest::Array{Int, 2} = Array{Int, 2}(undef, n_cutest, 2) # 1: nvar, 2: ncon
	names_cutest::Vector{String} = Vector{String}(undef, n_cutest)
	time_cutest::Array{Real, 3} = Array{Real, 3}(undef, n_solver, n_cutest, 5) # (n_solver rows, n_cutest cols, 2 in depth) (pb, solver, 1): neval_obj, (pb, solver, 2): neval_con
	resol_cutest::Array{GenericExecutionStats, 2} = Array{GenericExecutionStats, 2}(undef, n_solver, n_cutest)
	kkt_cutest::Array{Dict{String,Any}, 2} = Array{Dict{String,Any}, 2}(undef, n_solver, n_cutest)

	info_nlp::Array{Int, 2} = Array{Int, 2}(undef, n_nlp, 2) # 1: nvar, 2: ncon
	names_nlp::Vector{String} = Vector{String}(undef, n_nlp)
	time_nlp::Array{Real, 3} = Array{Real, 3}(undef, n_solver, n_nlp, 5) # (n_solver rows, n_nlp cols, 2 in depth) (pb, solver, 1): neval_obj, (pb, solver, 2): neval_con
	resol_nlp::Array{GenericExecutionStats, 2} = Array{GenericExecutionStats, 2}(undef, n_solver, n_nlp)
	kkt_nlp::Array{Dict{String,Any}, 2} = Array{Dict{String,Any}, 2}(undef, n_solver, n_nlp)

	info_ampl::Array{Int, 2} = Array{Int, 2}(undef, n_ampl, 2) # 1: nvar, 2: ncon
	names_ampl::Vector{String} = Vector{String}(undef, n_ampl)
	time_ampl::Array{Real, 3} = Array{Real, 3}(undef, n_solver, n_ampl, 5) # (n_solver rows, n_nlp cols, 2 in depth) (pb, solver, 1): neval_obj, (pb, solver, 2): neval_con
	resol_ampl::Array{GenericExecutionStats, 2} = Array{GenericExecutionStats, 2}(undef, n_solver, n_ampl)
	kkt_ampl::Array{Dict{String,Any}, 2} = Array{Dict{String,Any}, 2}(undef, n_solver, n_ampl)


	#** I. CUTEst problem set

	k = 0
	for i in cutest_pb_index_set
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
																															linear_residuals = linear_residuals,
																															KKT_checking = false,
																															warm_start = true)

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

				resol_cutest[i, k] = resol_nclres
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
																															linear_residuals = linear_residuals,
																															KKT_checking = true,

																															warm_start = true)

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

				resol_cutest[i, k] = resol_nclkkt
			end

			if solver[i] == "ipopt"
				reset!(nlp.counters)
				resol_solver, time_cutest[i, k, 3], time_cutest[i, k, 4], time_cutest[i, k, 5], memallocs = @timed NLPModelsIpopt.ipopt(nlp ; max_iter = max_iter_solver,
																																		tol = tol,
																																		constr_viol_tol = constr_viol_tol,
																																		compl_inf_tol = compl_inf_tol,
																																		print_level = print_level_iter,
																																		ignore_time = true)


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

				resol_cutest[i, k] = resol_solver
			end
		end
		finalize(nlp)
	end


	#** II. NLP problem set

	k = 0
	for i in nlp_pb_index_set
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
																													linear_residuals = linear_residuals,
																													KKT_checking = false,
																													warm_start = true)

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

				resol_nlp[i, k] = resol_nclres
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
																													linear_residuals = linear_residuals,
																													KKT_checking = true,
																													warm_start = true)

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

				resol_nlp[i, k] = resol_nclkkt
			end

			if solver[i] == "ipopt"
				reset!(nlp.counters)
				resol_solver, time_nlp[i, k, 3], time_nlp[i, k, 4], time_nlp[i, k, 5], memallocs = @timed NLPModelsIpopt.ipopt(nlp ; 
																																max_iter = max_iter_solver,
																																tol = tol,
																																print_level = print_level_iter,
																																constr_viol_tol = constr_viol_tol,
																																compl_inf_tol = compl_inf_tol,
																																ignore_time = true)

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

				resol_nlp[i, k] = resol_solver
			end
		end

	end


	#** III. AMPL problem set

	k = 0
	for i in ampl_pb_index_set
		k += 1
    
		#** III.1 Problem
		tax_name = ampl_pb_set[i]
		cd("/home/perselie/Bureau/projet/AMPL_tests/TAX")
    
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
																													linear_residuals = linear_residuals,
																													KKT_checking = false,
																													warm_start = true)

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

				resol_ampl[i, k] = resol_nclres
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
																													linear_residuals = linear_residuals,
																													KKT_checking = true,
																													warm_start = true)

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

				resol_ampl[i, k] = resol_nclkkt
			end

			if solver[i] == "ipopt"
				reset!(ampl_model.counters)
				resol_solver, time_ampl[i, k, 3], time_ampl[i, k, 4], time_ampl[i, k, 5], memallocs = @timed NLPModelsIpopt.ipopt(ampl_model ;
																													max_iter = max_iter_solver,
																													tol = tol,
																													constr_viol_tol = constr_viol_tol,
																													compl_inf_tol = compl_inf_tol,
																													print_level = print_level_iter,
																													ignore_time = true)

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

				resol_ampl[i, k] = resol_solver
			end
		end

		finalize(ampl_model)
		rm(tax_name * ".nl")

	end
  
	cd("/home/perselie/Bureau/projet/ncl/")

	#** IV. Data frames
	info = vcat(info_cutest, info_nlp, info_ampl)
	names = vcat(names_cutest, names_nlp, names_ampl)

	resol = hcat(resol_cutest, resol_nlp, resol_ampl)
	kkt = hcat(kkt_cutest, kkt_nlp, kkt_ampl)
	time = hcat(time_cutest, time_nlp, time_ampl)

	n_pb = n_cutest + n_nlp + n_ampl

	stats = Dict(Symbol(solver[i]) => DataFrame(:problem	=> [names[k] for k in 1:n_pb],
												:id			=> [k for k in 1:n_pb],
												:nvar 		=> [info[k, 1] for k in 1:n_pb],
												:ncon		=> [info[k, 2] for k in 1:n_pb],

												:niter 		=> [resolution.iter for resolution in resol[i, :]],
												:f 			=> [resolution.objective for resolution in resol[i, :]],

												:feval 		=> [time[i, k, 1] for k in 1:n_pb],
												:ceval 		=> [time[i, k, 2] for k in 1:n_pb],
												:time		=> [time[i, k, 3] for k in 1:n_pb],
												:bytes		=> [time[i, k, 4] for k in 1:n_pb],
												:gctime		=> [time[i, k, 5] for k in 1:n_pb],

												:feas 		=> [kkt_res["primal_feas"] for kkt_res in kkt[i, :]],
												:compl 		=> [kkt_res["complementarity_feas"] for kkt_res in kkt[i, :]],
												:mult_norm 	=> [norm(vcat(resolution.solver_specific[:multipliers_con], (resolution.solver_specific[:multipliers_L] - resolution.solver_specific[:multipliers_U])), Inf) for resolution in resol[i, :]],
												:lag_norm 	=> [kkt_res["dual_feas"] for kkt_res in kkt[i, :]],
												:r_norm 	=> [haskey(resolution.solver_specific, :residuals) ? norm(resolution.solver_specific[:residuals], Inf) : 0. for resolution in resol[i, :]],

												:solve_succeeded => [Symbol(resolution.solver_specific[:internal_msg]) for resolution in resol[i, :]],
												:r_opti 	=> [Symbol(haskey(resolution.solver_specific, :residuals) ? (norm(resolution.solver_specific[:residuals], Inf) <= tol) : true) for resolution in resol[i, :]],
												:r_acc_opti	=> [Symbol(haskey(resolution.solver_specific, :residuals) ? (norm(resolution.solver_specific[:residuals], Inf) <= acc_factor * tol) : true) for resolution in resol[i, :]],
												:kkt_opti 	=> [Symbol(kkt_res["optimal"]) for kkt_res in kkt[i, :]],
												:kkt_acc_opti => [Symbol(kkt_res["acceptable"]) for kkt_res in kkt[i, :]])
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

  
	ltx_file = open("./res/ltx_table.tex", write = true)
	latex_table(ltx_file, df_res)
	close(ltx_file)

	solved(df) = (df.solve_succeeded .== Symbol("Solve_Succeeded"))
	kkt_opti(df) = (df.kkt_opti .== Symbol(true))
	kkt_acc_opti(df) = (df.kkt_acc_opti .== Symbol(true))
	compare = [df -> df.f, df -> df.feval + df.ceval, df -> .!kkt_opti(df) * 1000 + df.feval, df -> .!kkt_acc_opti(df) * 1000 + df.feval]

	#println([stats[Symbol("ipopt")].f, .!solved(stats[Symbol("ipopt")]) * 10 + stats[Symbol("ipopt")].feval, .!kkt_opti(stats[Symbol("ipopt")]) * 10 + stats[Symbol("ipopt")].feval, .!kkt_acc_opti(stats[Symbol("ipopt")]) * 10 + stats[Symbol("ipopt")].feval])

	compare_names = ["Obj value", "Succeeded + f_eval", "Optimal KKT + f_eval", "Acceptable KKT + f_eval"]
	p = profile_solvers(stats, compare, compare_names)
	Plots.svg(p, "profilewall1000")

	return nothing
end
