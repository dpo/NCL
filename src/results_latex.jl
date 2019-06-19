"""
#################
res_tabular function
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



"""
Create a latex file which contains an array with the results of the ../res folder.
Each subfolder of the ../res folder contains the results of a resolution method.

Arguments
- outputFile: path of the output file

Prerequisites:
- Each subfolder must contain text files
- Each text file correspond to the resolution of one instance
- Each text file contains a variable "solveTime" and a variable "isOptimal"
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
    \usepackage{multicol}
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

    header *= "cccccccc}\n\t\\hline\n" #seven columns

    # column names
    header *= "\\\\\n\\textbf{Problem}  & \\textbf{\$n_{var}\$} & \\textbf{\$n_{con}\$} & \\textbf{\$n_{iter}\$} & \\textbf{\$f\\left(x\\right)\$} & \\textbf{\$\\left\\Vert r \\right\\Vert_\\infty\$} & \\textbf{\$\\left\\Vert \\nabla_x L \\right\\Vert_\\infty\$} & \\textbf{\$\\left\\Vert r \\right\\Vert_\\infty \\leq \\eta\$ ?} & \\textbf{Fits KKT ?} "

    header *= "\\\\\\hline\n"

    footer = raw"""\hline\end{tabular}
    \end{center}

    """
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

        println(fout, name * " & ", nvar, " & ", ncon, " & ", iter, " & ", obj_val, " & ", norm_r, " & ", norm_lag_grad, " & ")

        if optimal_res
            println(fout, "\$\\checkmark\$", " & ")
        else
            println(fout, "\$\\times\$", " & ")
        end
            
        if optimal_kkt
            println(fout, "\$\\checkmark\$")
        else
            println(fout, "\$\\times\$")
        end

        println(fout, "\\\\")

        id += 1
    end

    # Print the end of the latex file
    println(fout, footer)

    println(fout, "\\end{document}")

    close(fout)
    
end 
