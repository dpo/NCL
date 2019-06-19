using NLPModels
using CUTEst
using NLPModelsIpopt
using Printf
include("NCLSolve.jl")

#* hand made hs13 model
	f(x) = (x[1] - 2.) ^ 2  +  x[2] ^ 2
	c(x) = [(1-x[1]) ^ 3  -  x[2]]
	lcon = [0.]
	ucon = [Inf]
	lvar = [0., 0.]
	uvar = [Inf, Inf]
	x0 = [-2.,-2.]

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
		return [2.0 * obj_weight - 6 * y[1] * (1-x[1]) 0.0; 0.0 2 * obj_weight]
	end
		
	function NLPModels.hess_coord(nlp :: hs13, x :: AbstractVector; obj_weight=1.0, y=[0.])
		increment!(nlp, :neval_hess)
		return ([1,2], [1,2], [2.0 * obj_weight - 6 * y * (1-x[1]), 2. * obj_weight])
	end

	function NLPModels.hess_coord!(nlp :: hs13, x :: AbstractVector, hrows::Vector{<:Int64}, hcols::Vector{<:Int64}, hvals::Vector{<:Float64} ; obj_weight=1.0, y=[0.])
		increment!(nlp, :neval_hess)
		hvals[1] = 2.0 * obj_weight - 6 * y[1] * (1-x[1])
		hvals[2] = 2 * obj_weight
		return (hrows, hcols, hvals)
	end

	function NLPModels.hess_structure(nlp::hs13)
		increment!(nlp, :neval_hess)
		return ([1,2], [1,2])
	end
		
	function NLPModels.hprod!(nlp :: hs13, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight=1.0, y=Float64[])
		increment!(nlp, :neval_hprod)
		Hv .= [(2.0 * obj_weight - 6 * y * (1-x[1])) * v[1] ; 2. * obj_weight * v[2]]
		return Hv
	end

	function NLPModels.cons(nlp :: hs13, x :: AbstractVector)
		increment!(nlp, :neval_cons)
		return c(x)
	end
		
	function NLPModels.cons!(nlp :: hs13, x :: AbstractVector, cx :: AbstractVector)
		increment!(nlp, :neval_cons)
		cx .= c(x)
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




function pb_try()
	##** HS13
		f(x) = (x[1] - 2.) ^ 2  +  x[2] ^ 2
		c(x) = [(1-x[1]) ^ 3  -  x[2]]
		lcon = [0.]
		ucon = [Inf]
		lvar = [0., 0.]
		uvar = [Inf, Inf]
		x0 = [-2.,-2.]

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




	##** TAX1
	#nlp = CUTEstModel("TAX13322") # gros
	#nlp = CUTEstModel("TAX53322") # tres gros
	#nlp = CUTEstModel("TAX213322") # enorme

	ncl = NCLModel(nlp, res_lin_cons = false)

	resolution = NCLSolve(nlp ;
						max_iter_NCL = 20,
						max_iter_solver = 1000,
						print_level_NCL = 2,
						print_level_solver = 0,
						linear_residuals = true,
						output_file_print_NCL = true,
						output_file_print_solver = false,
						output_file_name_NCL = "NCL_on_hs13",
						warm_start_init_point = "yes")

	println(resolution.solver_specific[:internal_msg])

	finalize(nlp)
	return resolution.solver_specific[:internal_msg]
end

function pb_set_resolution( ; #No arguments, only key-word arguments
							path_res_folder::String = "/home/perselie/Bureau/projet/ncl/res/", 
							cutest_generic_pb_name::String = "CUTEst_HS", 
							cutest_pb_set::Vector{String} = ["HS$i" for i in [1,2,3,4,13,15,19,20,21]], 
							cutest_pb_index_set::Vector{Int64} = [i for i in 1:length(cutest_pb_set)], 
							nlp_generic_pb_name::String = "NLP_HS", 
							nlp_pb_set::Vector{<:AbstractNLPModel} = [hs13()],
							nlp_pb_index_set::Vector{Int64} = [1], 
							generate_latex::Bool = true,
							tol::Float64 = 1e-8,
							constr_viol_tol::Float64 = 1e-6,
							compl_inf_tol::Float64 = 1e-4,
							KKT_checking::Bool = false,
							linear_residuals = true
						  )

	

	#** I. CUTEst problem set
		#** I.0 Directory check
			if isdir(path_res_folder * cutest_generic_pb_name * "/")
				file_cutest = open(path_res_folder * cutest_generic_pb_name * "/" * cutest_generic_pb_name * ".log", write=true)
			else
				mkdir(path_res_folder * cutest_generic_pb_name * "/")
				file_cutest = open(path_res_folder * cutest_generic_pb_name * "/" * cutest_generic_pb_name * ".log", write=true)
			end

		for i in cutest_pb_index_set
			#** II.1 Problem
				pb = cutest_pb_set[i]
				nlp = CUTEstModel(pb)

			#** II.2 Resolution
				resol = NCLSolve(nlp ;
						max_iter_NCL = 20,
						tol = tol/10,
						constr_viol_tol = constr_viol_tol/10,
						compl_inf_tol = compl_inf_tol/10,
						max_iter_solver = 1000,
						print_level_NCL = 6,
						print_level_solver = 0,
						linear_residuals = linear_residuals,
						KKT_checking = KKT_checking,
						output_file_print_NCL = true,
						output_file_print_solver = false,
						output_file_NCL = file_cutest,
						warm_start_init_point = "yes")

				@printf(file_cutest, "\n=================\n")

				D = KKT_check(nlp, 
							  resol.solution, 
							  resol.solver_specific[:multipliers_con], 
							  resol.solver_specific[:multipliers_U], 
							  resol.solver_specific[:multipliers_L] ; 
							  tol = tol/10,
							  constr_viol_tol = constr_viol_tol/10,
							  compl_inf_tol = compl_inf_tol/10,
							  print_level = 3, 
							  output_file_print = true,
							  output_file = file_cutest
							 )
			
			
			#** II.3 Print summary


				summary_path = path_res_folder * cutest_generic_pb_name * "/summary_" * cutest_generic_pb_name * "_" * nlp.meta.name * ".txt"
				file_summary = open(summary_path, write=true)
				@printf(file_summary, "name = \"%s\" \n", nlp.meta.name)
				@printf(file_summary, "nvar = %d\n", nlp.meta.nvar)
				@printf(file_summary, "ncon = %d\n", nlp.meta.ncon)
				@printf(file_summary, "iter = %d\n", resol.iter)
				@printf(file_summary, "obj_val = %9.2e\n", resol.objective)
				@printf(file_summary, "norm_lag_grad = %9.2e\n", (nlp.meta.ncon != 0) ? norm(grad(nlp, resol.solution) - jtprod(nlp, resol.solution, resol.solver_specific[:multipliers_con]) - (resol.solver_specific[:multipliers_L]  - resol.solver_specific[:multipliers_U]), Inf) 
																					  : norm(grad(nlp, resol.solution) - (resol.solver_specific[:multipliers_L] - resol.solver_specific[:multipliers_U])[1:nlp.meta.nvar], Inf))
				@printf(file_summary, "norm_r = %9.2e\n", haskey(resol.solver_specific, :residuals) ? norm(resol.solver_specific[:residuals]) : 0.)
				@printf(file_summary, "optimal_res = %s\n", haskey(resol.solver_specific, :residuals) ? (norm(resol.solver_specific[:residuals]) <= constr_viol_tol) : true)
				@printf(file_summary, "optimal_kkt = %s\n", D["optimal"])
				@printf(file_summary, "acceptable_kkt = %s\n", D["acceptable"])


				close(file_summary)
			
			
			@printf(file_cutest, "\n============= End of resolution =============\n\n\n\n\n")
			finalize(nlp)
		end

		close(file_cutest)



	#** II. NLP problem set
		#** I.0 Directory check
			if isdir(path_res_folder * nlp_generic_pb_name * "/")
				file_nlp = open(path_res_folder *  nlp_generic_pb_name * "/" * nlp_generic_pb_name * ".log", write=true)
			else
				mkdir(path_res_folder * nlp_generic_pb_name * "/")
				file_nlp = open(path_res_folder *  nlp_generic_pb_name * "/" * nlp_generic_pb_name * ".log", write=true)
			end

		for i in nlp_pb_index_set
			#** II.1 Problem
				pb = nlp_generic_pb_name * string(i)
				nlp = nlp_pb_set[i]

			#** II.2 Resolution
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
						warm_start_init_point = "yes")

				@printf(file_nlp, "\n=================\n")


				D = KKT_check(nlp, 
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
				@printf(file_summary, "name = \"%s\" \n", nlp.meta.name)
				@printf(file_summary, "nvar = %d\n", nlp.meta.nvar)
				@printf(file_summary, "ncon = %d\n", nlp.meta.ncon)
				@printf(file_summary, "iter = %d\n", resol.iter)
				@printf(file_summary, "obj_val = %7.2e\n", resol.objective)
				@printf(file_summary, "norm_lag_grad = %7.2e\n", (nlp.meta.ncon != 0) ? norm(grad(nlp, resol.solution) - jtprod(nlp, resol.solution, resol.solver_specific[:multipliers_con]) - (resol.solver_specific[:multipliers_L]  - resol.solver_specific[:multipliers_U]), Inf) 
																					  : norm(grad(nlp, resol.solution) - (resol.solver_specific[:multipliers_L] - resol.solver_specific[:multipliers_U])[1:nlp.meta.nvar], Inf))
				@printf(file_summary, "norm_r = %7.2e\n", haskey(resol.solver_specific, :residuals) ? norm(resol.solver_specific[:residuals]) : 0.)
				@printf(file_summary, "optimal_res = %s\n", haskey(resol.solver_specific, :residuals) ? (norm(resol.solver_specific[:residuals]) <= constr_viol_tol) : true)
				@printf(file_summary, "optimal_kkt = %s\n", D["optimal"])
				@printf(file_summary, "acceptable_kkt = %s\n", D["acceptable"])

				close(file_summary)


			@printf(file_nlp, "\n============= End of resolution =============\n\n\n\n\n")
		
		end

		close(file_nlp)

	if generate_latex
		res_tabular("../res/latex.tex")
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

    \begin{document}""")

        header = raw"""
    \begin{center}
	\renewcommand{\arraystretch}{1.4} 
	\begin{landscape}
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

    header *= "ccccccccc}\n\t\\hline\n" #seven columns

    # column names
    header *= "\\\\\n\\textbf{Problem}  & \\textbf{\$n_{var}\$} & \\textbf{\$n_{con}\$} & \\textbf{\$n_{iter}\$} & \\textbf{\$f\\left(x\\right)\$} & \\textbf{\$\\left\\Vert r \\right\\Vert_\\infty\$} & \\textbf{\$\\left\\Vert \\nabla_x L \\right\\Vert_\\infty\$} & \\textbf{\$\\left\\Vert r \\right\\Vert_\\infty \\leq \\eta\$ ?} & \\textbf{KKT\\checkmark ?} & \\textbf{Almost KKT\\checkmark ?} "

    header *= "\\\\\\hline\n"

    footer = raw"""\hline\end{tabular}\end{landscape}
	\end{center}"""
	
    println(fout, header)

    # On each page an array will contain at most maxInstancePerPage lines with results
    maxInstancePerPage = 30
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

        println(fout, name * " & ", nvar, " & ", ncon, " & ", iter, " & ", obj_val, " & ", norm_r, " & ", norm_lag_grad, " & ", optimal_res ? "\\color{green} \\checkmark" : "\\color{red} \$\\times\$", " & ", optimal_kkt ? "\\color{green} \\checkmark" : "\\color{red} \$\\times\$", " & ", acceptable_kkt ? "\\color{green} \\checkmark" : "\\color{red} \$\\times\$")

        println(fout, "\\\\")

        id += 1
    end

    # Print the end of the latex file
    println(fout, footer)

    println(fout, "\\end{document}")

    close(fout)
    
end 








#pb_set_resolution(generate_latex = true)
res_tabular("../res/latex.tex")