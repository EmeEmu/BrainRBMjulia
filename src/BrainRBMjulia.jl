module BrainRBMjulia

import CUDA
import HDF5
using RestrictedBoltzmannMachines: RBM, StandardizedRBM, 
  Binary, xReLU, pReLU, Gaussian,
  pcd!, initialize!, standardize,
  sample_from_inputs, sample_h_from_v, sample_v_from_h, sample_v_from_v,
  mean_from_inputs, mode_from_inputs, mean_h_from_v, mean_v_from_h,
  inputs_h_from_v, inputs_v_from_h,
  free_energy,  log_pseudolikelihood,
  AbstractLayer, var_from_inputs, gpu, cpu,
  save_rbm, load_rbm
export sample_from_inputs, sample_h_from_v, sample_v_from_h, sample_v_from_v,
  mean_from_inputs, mode_from_inputs, mean_h_from_v, mean_v_from_h,
  free_energy, log_pseudolikelihood, var_from_inputs,
  gpu, cpu,
  save_rbm, load_rbm
  
using Statistics: mean, std, var, cov, cor, quantile
export mean, std, var, cov, cor, quantile

using Random
using ProgressLogging: @progress, ProgressLevel
using Logging: @logmsg
using UUIDs: uuid4
using Clustering
using AbstractFFTs
using FLoops
using Optimisers: Adam, adjust!
using ValueHistories: MVHistory, @trace
using Combinatorics
using HDF5: h5open, create_group, read, attrs


include("rbms.jl")
export BrainRBM, StateRBM, build_training_h, translate, reconstruct

include("train.jl")
export training_wrapper, training_wrapper_srbm

include("utils/dataset.jl")
export Data

include("utils/misc.jl")
export zscore, quantile_2d

include("utils/split_testtrain.jl")
export split_set, DatasetSplit, SectionSplit, section_moments

include("utils/voxels.jl")
export VoxelGrid, vox_to_neur_activity

include("utils/reorder.jl")
export reorder_hus!, linear_order, reorder_states_corr

include("utils/swapsign.jl")
export swap_hidden_sign, swap_hidden_sign!

include("generate.jl")
export gen_data, GeneratedData

include("statistics.jl")
export compute_all_moments, reconstruction_likelihood, MomentsAggregate

include("utils/nRMSE.jl")
export nRMSE, nRMSE_from_moments, nRMSEs_Lp, nRMSEs_L4

include("utils/coupling.jl")
export coupling_approx

include("utils/binary_utils.jl")
export encode_binary, decode_binary, Nstates_per_Nbits, Nbits_per_Nstates

include("state_sampling.jl")
export state_sampling, state_distrib, state_proba, state_transition, mean_v_by_s, state_max

include("maps.jl")
export BoxAround, JS_distance, create_map, map_finite!, interpolation, Maps, dump_maps

include("saving.jl")
export dump_data, load_data, dump_brainRBM, load_brainRBM, dump_stateRBM, load_stateRBM, load_brainRBM_eval, rank_brainRBMs, dump_voxel

end
