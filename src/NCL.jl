module NCLSolve



# comment
#** Important
# ! Warning / Problem
# ? Question
# TODO

using Ipopt

using LinearAlgebra
using Printf

using NLPModels
using SolverTools
using NLPModelsIpopt: ipopt

include("NCLModel.jl")
include("KKT_check.jl")
include("NCLSolve.jl")

#using NLPModelsKnitro

#! TODO Fix closing file problem...

######### TODO #########
######### TODO #########
######### TODO #########

    # TODO (feature)   : Créer un vrai statut
    # TODO (infos)     : Lecture des infos de ipopt
    # TODO KKT_check output in file to fix

    # TODO (recherche) : choix des mu_init à améliorer...
    # TODO (recherche) : Points intérieurs à chaud...
    # TODO (recherche) : tester la proximité des multiplicateurs y_k de renvoyés par le solveur et le ncl.y du problème (si r petit, probablement proches.)
    # TODO (recherche) : Mieux choisir le pas pour avoir une meilleure convergence
    # TODO (recherche) : ajuster eta_end

    # TODO (Plus tard) : Pierric pour choix de alpha, beta, tau...
########## TODO ########
########## TODO ########
########## TODO ########


end #end of module
