"""
    str_type(obj)

Return the base type name of `obj` without type parameters. Useful when
storing objects in HDF5 attributes.
"""
function str_type(obj::Any)
  return split(string(typeof(obj)), "{")[1]
end

"""
    c(x::BitMatrix)
    c(x::Matrix)

Convert matrices to a storage-friendly format. Bit matrices are converted to
`UInt8` while other matrices are returned unchanged.
"""
function c(x::BitMatrix)
  return UInt8.(x)
end
function c(x::Matrix)
  return x
end

"""
    uc(x::Matrix{UInt8})
    uc(x::Matrix)

Inverse operation of [`c`](@ref). Convert matrices read from disk back to
their logical type.
"""
function uc(x::Matrix{UInt8})
  return Bool.(x)
end
function uc(x::Matrix)
  return x
end

"""
    layer_to_hdf5Group(grp::HDF5.Group, layer::AbstractLayer)

Store layer parameters and metadata into the HDF5 group `grp`.
"""
function layer_to_hdf5Group(grp::HDF5.Group, layer::AbstractLayer)
  attrs(grp)["type"] = str_type(layer)
  attrs(grp)["size"] = string(size(layer))
  grp["params"] = layer.par
end

"""
    rbm_to_hdf5Group(rbm_grp::HDF5.Group, rbm; comment::String="")

Serialize `rbm` into the group `rbm_grp`. Works for both [`RBM`] and
[`StandardizedRBM`] models.
"""
function rbm_to_hdf5Group(rbm_grp::HDF5.Group, rbm::StandardizedRBM; comment::String="")
  attrs(rbm_grp)["type"] = str_type(rbm)
  attrs(rbm_grp)["comment"] = comment

  v_grp = create_group(rbm_grp, "VisibleLayer")
  layer_to_hdf5Group(v_grp, rbm.visible)

  h_grp = create_group(rbm_grp, "HiddenLayer")
  layer_to_hdf5Group(h_grp, rbm.hidden)

  rbm_grp["weights"] = rbm.w

  stand_grp = create_group(rbm_grp, "Standardization")
  stand_grp["offset_h"] = rbm.offset_h
  stand_grp["offset_v"] = rbm.offset_v
  stand_grp["scale_h"] = rbm.scale_h
  stand_grp["scale_v"] = rbm.scale_v
end;
function rbm_to_hdf5Group(rbm_grp::HDF5.Group, rbm::RBM; comment::String="")
  attrs(rbm_grp)["type"] = str_type(rbm)
  attrs(rbm_grp)["comment"] = comment

  v_grp = create_group(rbm_grp, "VisibleLayer")
  layer_to_hdf5Group(v_grp, rbm.visible)

  h_grp = create_group(rbm_grp, "HiddenLayer")
  layer_to_hdf5Group(h_grp, rbm.hidden)

  rbm_grp["weights"] = rbm.w
end;

# layer_type(::Binary) = "Binary"
# layer_type(::Gaussian) = "Gaussian"
# layer_type(::xReLU) = "xReLU"
# layer_type(::RestrictedBoltzmannMachines.Binary) = "Binary"
# layer_type(::RestrictedBoltzmannMachines.Gaussian) = "Gaussian"
# layer_type(::RestrictedBoltzmannMachines.xReLU) = "xReLU"

# construct_layer(layer_type::AbstractString, par::AbstractArray) = construct_layer(Val(Symbol(layer_type)), par)

# construct_layer(::Val{:Binary}, par::AbstractArray) = Binary(par)
# construct_layer(::Val{:Gaussian}, par::AbstractArray) = Gaussian(par)
# construct_layer(::Val{:xReLU}, par::AbstractArray) = xReLU(par)
# construct_layer(AbstractString{"RestrictedBoltzmannMachines.Binary"}, par::AbstractArray) = Binary(par)
# construct_layer(AbstractString{"RestrictedBoltzmannMachines.Gaussian"}, par::AbstractArray) = Gaussian(par)
# construct_layer(AbstractString{"RestrictedBoltzmannMachines.xReLU"}, par::AbstractArray) = xReLU(par)
"""
    construct_layer(layer_type::AbstractString, par::AbstractArray)

Rebuild a layer from its string type name and parameter array.
"""
function construct_layer(layer_type::AbstractString, par::AbstractArray)
  if occursin("Binary", layer_type)
    return Binary(par)
  elseif occursin("xReLU", layer_type)
    return xReLU(par)
  elseif occursin("Gaussian", layer_type)
    return Gaussian(par)
  else
    throw("layer type $(layer_type) not recognized.")
  end
end

"""
    rbm_from_hdf5Group(grp::HDF5.Group)

Reconstruct an [`RBM`] or [`StandardizedRBM`] previously stored with
[`rbm_to_hdf5Group`](@ref).
"""
function rbm_from_hdf5Group(grp::HDF5.Group)
  # vlayer_type = attrs(grp["VisibleLayer"])["type"]
  # v_layer = getfield(BrainRBMjulia, Symbol(vlayer_type))(read(grp["VisibleLayer/params"]))
  v_layer = construct_layer(attrs(grp["VisibleLayer"])["type"], read(grp["VisibleLayer/params"]))

  # hlayer_type = attrs(grp["HiddenLayer"])["type"]
  # h_layer = getfield(BrainRBMjulia, Symbol(hlayer_type))(read(grp["HiddenLayer/params"]))
  h_layer = construct_layer(attrs(grp["HiddenLayer"])["type"], read(grp["HiddenLayer/params"]))

  rbm_type = attrs(grp)["type"]
  if occursin("StandardizedRBM", rbm_type)
    rbm = StandardizedRBM(
      v_layer,
      h_layer,
      read(grp["weights"]),
      read(grp["Standardization/offset_v"]),
      read(grp["Standardization/offset_h"]),
      read(grp["Standardization/scale_v"]),
      read(grp["Standardization/scale_h"]),
    )
  else
    rbm = RBM(
      v_layer,
      h_layer,
      read(grp["weights"]),
    )
  end
  return rbm
end

"""
    dic_to_hdf5Group(grp::HDF5.Group, dic::Dict; comment::String="")

Write all key-value pairs of `dic` into `grp`, storing `comment` as an
attribute.
"""
function dic_to_hdf5Group(grp::HDF5.Group, dic::Dict; comment::String="")
  attrs(grp)["comment"] = comment

  for k in keys(dic)
    grp[k] = dic[k]
  end
end

"""
    dic_from_hdf5Group(grp::HDF5.Group) -> Dict

Read all datasets of `grp` into a dictionary.
"""
function dic_from_hdf5Group(grp::HDF5.Group)
  dic = Dict()

  for k in keys(grp)
    dic[k] = grp[k][]
  end
  return dic
end

"""
    datasplit_to_hdf5Group(grp::HDF5.Group, split::DatasetSplit; comment::String="")

Store a [`DatasetSplit`] in the group `grp`.
"""
function datasplit_to_hdf5Group(grp::HDF5.Group, split::DatasetSplit; comment::String="")
  attrs(grp)["comment"] = comment

  grp["train"] = c(split.train)
  grp["valid"] = c(split.valid)
  grp["train_inds"] = split.train_inds
  grp["valid_inds"] = split.valid_inds
end

"""
    datasplit_from_hdf5Group(grp::HDF5.Group) -> DatasetSplit

Load a [`DatasetSplit`] previously saved with
[`datasplit_to_hdf5Group`](@ref).
"""
function datasplit_from_hdf5Group(grp::HDF5.Group)
  return DatasetSplit(
    grp["train_inds"][],
    grp["valid_inds"][],
    uc(grp["train"][]),
    uc(grp["valid"][]),
  )
end

"""
    generated_to_hdf5Group(grp::HDF5.Group, gen::GeneratedData; comment::String="")

Write a [`GeneratedData`] object to `grp`.
"""
function generated_to_hdf5Group(grp::HDF5.Group, gen::GeneratedData; comment::String="")
  attrs(grp)["comment"] = comment

  grp["v"] = c(gen.v)
  grp["h"] = c(gen.h)
  grp["thermal"] = gen.thermal
  grp["nsamples"] = gen.nsamples
  grp["nstep"] = gen.nstep
  grp["nthermal"] = gen.nthermal
end

"""
    generated_from_hdf5Group(grp::HDF5.Group) -> GeneratedData

Reconstruct [`GeneratedData`] from the datasets stored in `grp`.
"""
function generated_from_hdf5Group(grp::HDF5.Group)
  return GeneratedData(
    uc(grp["v"][]),
    uc(grp["h"][]),
    grp["thermal"][],
    grp["nsamples"][],
    grp["nstep"][],
    grp["nthermal"][],
  )
end

"""
    dump_brainRBM(
        filename::String, 
        rbm::Union{RBM, StandardizedRBM}, 
        train_params::Dict, 
        evaluation::Dict, 
        split::DatasetSplit, 
        gen::GeneratedData,
        translated::Matrix; 
        comment::String=""
      )::String

Save all information about a trained RBM to a HDF5 file.

### Input

- `filename`    -- name of the file to save the RBM
- `rbm`         -- trained RBM
- `train_params`-- dictionary with training parameters
- `evaluation`  -- dictionary with evaluation of the rbm;
                    typicaly nRMSE values
- `split`       -- dataset split used for training
- `gen`         -- generated data
- `translated`  -- translated spikes into hidden activity
- `comment`     -- (optional, default: `""`) comment to add to the file

### Output

- `filename`    -- name of the file saved
"""
function dump_brainRBM(filename::String, rbm::Union{RBM,StandardizedRBM}, train_params::Dict, evaluation::Dict, split::DatasetSplit, gen::GeneratedData, translated::Matrix; comment::String="")
  fid = h5open(filename, "cw")
  grp = create_group(fid, "brainRBM")
  attrs(grp)["comment"] = comment

  rbm_grp = create_group(grp, "RBM")
  rbm_to_hdf5Group(rbm_grp, rbm)

  eval_grp = create_group(grp, "evaluation")
  dic_to_hdf5Group(eval_grp, evaluation)

  train_grp = create_group(grp, "train_params")
  dic_to_hdf5Group(train_grp, train_params)

  split_grp = create_group(grp, "Datasplit")
  datasplit_to_hdf5Group(split_grp, split)

  gen_grp = create_group(grp, "Generated")
  generated_to_hdf5Group(gen_grp, gen)

  grp["translated_spikes"] = c(translated)

  close(fid)
  return filename
end

"""
    load_brainRBM(
        filename::String
      )::Tuple{
          Union{RBM, StandardizedRBM}, 
          Dict, 
          Dict, 
          DatasetSplit, 
          GeneratedData, 
          Matrix
        }

Load all information about a trained RBM from a HDF5 file.

### Input

- `filename`    -- name of the file to load the RBM

### Output

- `rbm`         -- trained RBM
- `train_params`-- dictionary with training parameters
- `evaluation`  -- dictionary with evaluation of the rbm;
                    typicaly nRMSE values
- `split`       -- dataset split used for training
- `gen`         -- generated data
- `translated`  -- translated spikes into hidden activity
"""
function load_brainRBM(filename::String)
  fid = h5open(filename, "r")
  grp = fid["brainRBM"]

  rbm = rbm_from_hdf5Group(grp["RBM"])
  train_params = dic_from_hdf5Group(grp["train_params"])
  evaluation = dic_from_hdf5Group(grp["evaluation"])
  split = datasplit_from_hdf5Group(grp["Datasplit"])
  gen = generated_from_hdf5Group(grp["Generated"])
  translated = uc(grp["translated_spikes"][])

  close(fid)
  return rbm, train_params, evaluation, split, gen, translated
end

"""
    load_brainRBM_eval(filename::String; ignore::String="")::Dict
    load_brainRBM_eval(filenames::Vector{String}; ignore::String="")::Vector{Dict}

Load evaluation of one or multiple trained RBM(s) from HDF5 file(s). Typically nRMSE values.

### Input

- `filename` or `filenames` -- name of the file(s) to load the RBM(s)
- `ignore`      -- (optional, default: `""`) ignore a specific key

### Output

- `evaluation`  -- dictionary (or list of dictionaries) containing the evaluation(s) of the rbm(s);
                    typicaly nRMSE values
"""
function load_brainRBM_eval(filename::String; ignore="")
  fid = h5open(filename, "r")
  grp = fid["brainRBM"]
  evaluation = dic_from_hdf5Group(grp["evaluation"])
  close(fid)
  if ~isempty(ignore)
    delete!(evaluation, ignore)
  end
  return evaluation
end
function load_brainRBM_eval(filenames::Vector{String}; ignore="")
  return [load_brainRBM_eval(f; ignore=ignore) for f in filenames]
end
"""
    rank_brainRBMs(paths::Vector{String}; ignore="", bestn=-1)
    rank_brainRBMs(paths::Vector{String}, norm::Function; ignore="", bestn=-1)

Rank saved RBM models according to their evaluation metrics. The default
metric uses `nRMSEs_L4`; a custom `norm` function can be supplied. If
`bestn` is positive, only that many top-performing models are returned.
"""
function rank_brainRBMs(paths::Vector{String}; ignore="", bestn=-1)
  evals = load_brainRBM_eval(paths; ignore=ignore)
  norms = nRMSEs_L4(evals)
  inds = sortperm(norms)
  if bestn != -1
    inds = inds[1:bestn]
  end
  return paths[inds], norms[inds]
end
function rank_brainRBMs(paths::Vector{String}, norm::Function; ignore="", bestn=-1)
  evals = load_brainRBM_eval(paths; ignore=ignore)
  norms = norm(evals)
  inds = sortperm(norms)
  if bestn != -1
    inds = inds[1:bestn]
  end
  return paths[inds], norms[inds]
end


"""
    dump_stateRBM(
        filename::String, 
        rbmpath::String,
        srbm::RBM, 
        train_params::Dict, 
        evaluation::Dict, 
        split::DatasetSplit, 
        gen::GeneratedData,
        v_mean_std::Array{Float64, 3},
        state_proba::Matrix{Float64}; 
        comment::String=""
      )::String

Save all information about a trained RBM to a HDF5 file.

### Input

- `filename`    -- name of the file to save the RBM
- `rbmpath`     -- path to the parent brainRBM .h5 file
- `srbm`        -- trained stateRBM
- `train_params`-- dictionary with training parameters
- `evaluation`  -- dictionary with evaluation of the rbm;
                    typicaly nRMSE values
- `split`       -- dataset split used for training
- `gen`         -- generated data
- `v_mean_std`  -- mean and std of visible activation in the brainRBM
                    for each state (computed from repeated sampling)
- `state_proba` -- probability of each state for each time frame in
                    the data
- `comment`     -- (optional, default: `""`) comment to add to the file

### Output

- `filename`    -- name of the file saved
"""
function dump_stateRBM(
  filename::String,
  rbmpath::String,
  srbm::RBM,
  train_params::Dict,
  evaluation::Dict,
  split::DatasetSplit,
  gen::GeneratedData,
  v_mean_std::Array{Float64,3},
  state_proba::Matrix{Float64};
  comment::String=""
)
  fid = h5open(filename, "cw")
  grp = create_group(fid, "stateRBM")
  attrs(grp)["comment"] = comment
  grp["parent_rbm_path"] = rbmpath

  rbm_grp = create_group(grp, "RBM")
  rbm_to_hdf5Group(rbm_grp, srbm)

  eval_grp = create_group(grp, "evaluation")
  dic_to_hdf5Group(eval_grp, evaluation)

  train_grp = create_group(grp, "train_params")
  dic_to_hdf5Group(train_grp, train_params)

  split_grp = create_group(grp, "Datasplit")
  datasplit_to_hdf5Group(split_grp, split)

  gen_grp = create_group(grp, "Generated")
  generated_to_hdf5Group(gen_grp, gen)

  grp["visible_mean_std"] = v_mean_std
  grp["state_propability"] = state_proba

  close(fid)
  return filename
end


"""
    load_stateRBM(
        filename::String
      )::Tuple{
            String,
            RBM, 
            Dict, 
            Dict, 
            DatasetSplit, 
            GeneratedData,
            Array{Float64, 3},
            Matrix{Float64}; 
        }

Load all information about a trained stateRBM from a HDF5 file.

### Input

- `filename`    -- name of the file to load the RBM

### Output

- `rbmpath`     -- path to the parent brainRBM .h5 file
- `srbm`        -- trained stateRBM
- `train_params`-- dictionary with training parameters
- `evaluation`  -- dictionary with evaluation of the rbm;
                    typicaly nRMSE values
- `split`       -- dataset split used for training
- `gen`         -- generated data
- `v_mean_std`  -- mean and std of visible activation in the brainRBM
                    for each state (computed from repeated sampling)
- `state_proba` -- probability of each state for each time frame in
                    the data
"""
function load_stateRBM(filename::String)
  fid = h5open(filename, "r")
  grp = fid["stateRBM"]

  rbmpath = grp["parent_rbm_path"][]
  srbm = rbm_from_hdf5Group(grp["RBM"])
  train_params = dic_from_hdf5Group(grp["train_params"])
  evaluation = dic_from_hdf5Group(grp["evaluation"])
  split = datasplit_from_hdf5Group(grp["Datasplit"])
  gen = generated_from_hdf5Group(grp["Generated"])
  v_mean_std = grp["visible_mean_std"][]
  state_proba = grp["state_propability"][]

  close(fid)
  return rbmpath, srbm, train_params, evaluation, split, gen, v_mean_std, state_proba
end










"""
    dump_data(filename::String, data::Data; comment::String="")::String

Save the data used to train an RBM to a HDF5 file.

### Input

- `filename`    -- name of the file to save the data
- `data`        -- data used to train the RBM
- `comment`     -- (optional, default: `""`) comment to add to the file

### Output

- `filename`    -- name of the file saved
"""
function dump_data(filename::String, data::Data; comment::String="")
  fid = h5open(filename, "cw")
  grp = create_group(fid, "Data")
  attrs(grp)["comment"] = comment

  grp["name"] = data.name
  grp["coords"] = data.coords
  grp["spikes"] = c(data.spikes)

  close(fid)
  return filename
end

"""
    load_data(filename::String)::Data

Load the data used to train an RBM from a HDF5 file.

### Input

- `filename`    -- name of the file to load the data

### Output

- `data`        -- data used to train the RBM
"""
function load_data(filename::String)
  fid = h5open(filename, "r")
  grp = fid["Data"]

  name = grp["name"][]
  coords = grp["coords"][]
  spikes = uc(grp["spikes"][])

  close(fid)
  return Data(
    name,
    spikes,
    coords,
  )
end


"""
    dump_voxel(filename::String, voxel; comment::String="") -> String

Store a `voxel` grid object to `filename` in HDF5 format and return the
`filename`.
"""
function dump_voxel(filename::String, voxel::Any; comment::String="")
  fid = h5open(filename, "cw")
  grp = create_group(fid, "VoxelGrid")                                                       #----- MAIN GROUP 
  attrs(grp)["comment"] = comment

  grp["origins"] = voxel.origins
  grp["ends"] = voxel.ends
  grp["Ns"] = voxel.Ns
  grp["voxsize"] = voxel.voxsize
  grp["map"] = voxel.map
  grp["goods"] = voxel.goods


  voxel_activities = create_group(grp, "voxel_activities")                                   #----- 1 x Main Group [VoxelGrid]
  for i in 1:size(voxel.voxel_activities, 1)
    subgroup_name = "fish$i"
    subgroup = create_group(voxel_activities, subgroup_name)               #----- 2 x sub-Group [voxel_activities]
    subgroup["activities"] = voxel.voxel_activities[i]
  end



  neuron_affiliations = create_group(grp, "neuron_affiliations")                             #----- 1 x sub-Group [VoxelGrid]
  for i in 1:size(voxel.neuron_affiliation, 1)
    subgroup_name = "fish$i"
    subgroup = create_group(neuron_affiliations, subgroup_name)    #----- 2 x sub-Group [neuron_affiliations]
    subgroup["neuron_affiliation"] = voxel.neuron_affiliation[i]
  end



  voxel_compositions = create_group(grp, "voxel_compositions")                               #----- 1 x sub-Group [VoxelGrid]
  for i in 1:size(voxel.voxel_composition, 1)
    subgroup_name = "fish$i"
    subgroup = create_group(voxel_compositions, subgroup_name)                     #----- 2 x sub-Group [voxel_compositions]

    for j in 1:size(voxel.voxel_composition[i, :], 1)
      subsubgroup_name = "voxel$j"
      subsubgroup = create_group(subgroup, subsubgroup_name)                 #----- 3 x sub-Group [fish]
      subsubgroup["voxel$j"] = voxel.voxel_composition[i, j]
    end
  end



  close(fid)
  return filename
end
