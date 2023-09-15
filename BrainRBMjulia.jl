module BrainRBMjulia


using StandardizedRestrictedBoltzmannMachines: standardize, StandardizedRBM
using RestrictedBoltzmannMachines: RBM, Binary, xReLU, pReLU, Gaussian, pcd!, initialize!,
    sample_from_inputs, sample_h_from_v, sample_v_from_h, sample_v_from_v, mean_from_inputs,mode_from_inputs,
    mean_h_from_v, mean_v_from_h, free_energy, inputs_h_from_v, inputs_v_from_h, log_pseudolikelihood, AbstractLayer
using CudaRBMs: gpu, cpu
using Optimisers: Adam, adjust!
using Statistics: mean, std, var, cov, cor, quantile
using Random
using HDF5
using ValueHistories: MVHistory, @trace
using ProgressMeter
using Clustering

export gpu, cpu, Binary, xReLU, Gaussian, free_energy, RBM, StandardizedRBM, inputs_h_from_v, sample_from_inputs, RBM, translate, reconstruct, sample_h_from_v, AbstractLayer, mean_v_from_h, mean_h_from_v, initialize!, standardize
export mean, std, var, cov, cor, quantile


include("utils/dataset.jl")
export Data

include("utils/split_testtrain.jl")
export split_set, DatasetSplit

include("rbms.jl")
export BrainRBM

include("train.jl")
export training_wrapper

include("generate.jl")
export gen_data, GeneratedData

include("statistics.jl")
export compute_all_moments, reconstruction_likelihood, MomentsAggregate

include("saving.jl")
export dump_data, load_data, dump_brainRBM, load_brainRBM, load_brainRBM_eval, rank_brainRBMs


include("utils/nRMSE.jl")
export nRMSE, nRMSE_from_moments, nRMSEs_Lp, nRMSEs_L4

include("utils/reorder.jl")
export reorder_hus!

include("utils/swapsign.jl")
export swap_hidden_sign

end
