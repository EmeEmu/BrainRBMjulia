module BrainRBMjulia


using StandardizedRestrictedBoltzmannMachines: standardize, StandardizedRBM
using RestrictedBoltzmannMachines: RBM, Binary, xReLU, pReLU, Gaussian, pcd!, initialize!,
  sample_from_inputs, sample_h_from_v, sample_v_from_h, sample_v_from_v, mean_from_inputs, mode_from_inputs,
  mean_h_from_v, mean_v_from_h, free_energy, inputs_h_from_v, inputs_v_from_h, log_pseudolikelihood, AbstractLayer, var_from_inputs
using CudaRBMs: gpu, cpu
using Optimisers: Adam, adjust!
using Statistics: mean, std, var, cov, cor, quantile
using Random
using HDF5
using ValueHistories: MVHistory, @trace
using ProgressMeter
using Clustering
using LinearAlgebra: diagind
using FLoops
using NaNStatistics
using Combinatorics
using StatsBase: Histogram, fit

export gpu, cpu, Binary, xReLU, Gaussian, free_energy, RBM, StandardizedRBM, inputs_h_from_v, sample_from_inputs, RBM, translate, reconstruct, sample_h_from_v, AbstractLayer, mean_v_from_h, mean_h_from_v, initialize!, standardize
export mean, std, var, cov, cor, quantile


include("utils/dataset.jl")
export Data

include("utils/split_testtrain.jl")
export split_set, DatasetSplit, SectionSplit

include("utils/misc.jl")
export zscore, quantile_2d

include("utils/binary_utils.jl")
export encode_binary, decode_binary, Nstates_per_Nbits, Nbits_per_Nstates

include("rbms.jl")
export BrainRBM, StateRBM, build_training_h, translate, reconstruct
include("state_sampling.jl")
export state_sampling, state_distrib, state_proba, state_transition, mean_v_by_s, state_max

include("train.jl")
export training_wrapper, training_wrapper_srbm

include("generate.jl")
export gen_data, GeneratedData

include("statistics.jl")
export compute_all_moments, reconstruction_likelihood, MomentsAggregate

include("saving.jl")
export dump_data, load_data, dump_brainRBM, load_brainRBM, dump_stateRBM, load_stateRBM, load_brainRBM_eval, rank_brainRBMs, dump_voxel

include("maps.jl")
export smooth, BoxAround, build_map, JS_distance


include("utils/nRMSE.jl")
export nRMSE, nRMSE_from_moments, nRMSEs_Lp, nRMSEs_L4

include("utils/swapsign.jl")
export swap_hidden_sign

include("coupling.jl")
export coupling_approx

include("utils/voxels.jl")
export VoxelGrid, vox_to_neur_activity

include("utils/reorder.jl")
export reorder_hus!, linear_order, reorder_states_corr

end
