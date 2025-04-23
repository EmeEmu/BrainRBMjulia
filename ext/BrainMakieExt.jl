module BrainMakieExt

using Makie
import BrainRBMjulia
using Statistics: mean, std, var, cov, cor, quantile, median
using Clustering
using Colors
using ColorSchemes
using Interpolations

import BrainRBMjulia: dfsize, quantile_range
include("utils.jl")

import BrainRBMjulia: corrplotter, corrplotter!, couplingplotter, couplingplotter!
include("corr_plotter.jl")

import BrainRBMjulia: idplotter, idplotter!, multi_id
using BrainRBMjulia: nRMSE
include("id_plotter.jl")

import BrainRBMjulia: polarnrmseplotter, polarnrmseplotter!, multipolarnrmseplotter, multipolarnrmseplotter!
include("nrmse_polar.jl")

import BrainRBMjulia: rbmdiagram, rbmdiagram!
include("rbm_diagram.jl")

import BrainRBMjulia: colorscheme_alpha_sigmoid, colorscheme_whiteTOalpha_sigmoid, colorscheme_whiteTOalpha, cmap_aseismic, cmap_hardseismic, cmap_dff, cmap_Gbin, cmap_ainferno
include("colormaps.jl")

import BrainRBMjulia: neuron2dscatter, neuron2dscatter!, orthogonal_view_layout, neuronorthoscatter
include("brain_2D.jl")

using BrainRBMjulia: GeneratedData, DatasetSplit, free_energy, MomentsAggregate, nRMSE, xReLU
import BrainRBMjulia: generate_energy_plotter, stats_plotter, hu_params_plotter, hidden_hists
include("rbm_graphs.jl")

end
