function str_type(obj::Any)
    return split(string(typeof(obj)),"{")[1]
end

function c(x::BitMatrix)
    return UInt8.(x)
end
function c(x::Matrix)
    return x
end
function uc(x::Matrix{UInt8})
    return Bool.(x)
end
function uc(x::Matrix)
    return x
end

function layer_to_hdf5Group(grp::HDF5.Group, layer::AbstractLayer)
    attrs(grp)["type"] = str_type(layer)
    attrs(grp)["size"] = string(size(layer))
    grp["params"] = layer.par
end

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
    stand_grp["scale_v"] = rbm.scale_v;
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
function rbm_from_hdf5Group(grp::HDF5.Group)
    vlayer_type = attrs(grp["VisibleLayer"])["type"]
    v_layer = getfield(Main, Symbol(vlayer_type))(read(grp["VisibleLayer/params"]))

    hlayer_type = attrs(grp["HiddenLayer"])["type"]
    h_layer = getfield(Main, Symbol(hlayer_type))(read(grp["HiddenLayer/params"]))

    rbm_type = attrs(grp)["type"]
    if rbm_type == "StandardizedRBM"
        rbm = StandardizedRBM(
            v_layer, 
            h_layer, 
            read(grp["weights"]),
            read(grp["Standardization/offset_v"]),
            read(grp["Standardization/offset_h"]),
            read(grp["Standardization/scale_v"]),
            read(grp["Standardization/scale_h"]),
        )
    elseif rbm_type == "RBM"
        rbm = RBM(
            v_layer, 
            h_layer, 
            read(grp["weights"]),
        )
    end
    return rbm
end

function dic_to_hdf5Group(grp::HDF5.Group, dic::Dict; comment::String="")
    attrs(grp)["comment"] = comment
    
    for k in keys(dic)
        grp[k] = dic[k]
    end
end
function dic_from_hdf5Group(grp::HDF5.Group)
    dic = Dict()
    
    for k in keys(grp)
        dic[k] = grp[k][]
    end
    return dic
end

function datasplit_to_hdf5Group(grp::HDF5.Group, split::DatasetSplit; comment::String="")
    attrs(grp)["comment"] = comment
    
    grp["train"] = c(split.train)
    grp["valid"] = c(split.valid)
    grp["train_inds"] = split.train_inds
    grp["valid_inds"] = split.valid_inds
end
function datasplit_from_hdf5Group(grp::HDF5.Group)
    return DatasetSplit(
        grp["train_inds"][],
        grp["valid_inds"][],
        uc(grp["train"][]),
        uc(grp["valid"][]),
    )
end

function generated_to_hdf5Group(grp::HDF5.Group, gen::GeneratedData; comment::String="")
    attrs(grp)["comment"] = comment
    
    grp["v"] = c(gen.v)
    grp["h"] = c(gen.h)
    grp["thermal"] = gen.thermal
    grp["nsamples"] = gen.nsamples
    grp["nstep"] = gen.nstep
    grp["nthermal"] = gen.nthermal
end
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

function dump_brainRBM(filename::String, rbm::Union{RBM, StandardizedRBM}, train_params::Dict, split::DatasetSplit, gen::GeneratedData, translated::Matrix; comment::String="")
    fid = h5open(filename, "cw")
    grp = create_group(fid, "brainRBM")
    attrs(grp)["comment"] = comment
    
    rbm_grp = create_group(grp, "RBM")
    rbm_to_hdf5Group(rbm_grp, rbm)
    
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
function load_brainRBM(filename::String)
    fid = h5open(filename, "r")
    grp = fid["brainRBM"]
    
    rbm = rbm_from_hdf5Group(grp["RBM"])
    train_params = dic_from_hdf5Group(grp["train_params"])
    split = datasplit_from_hdf5Group(grp["Datasplit"])
    gen = generated_from_hdf5Group(grp["Generated"])
    translated = uc(grp["translated_spikes"][])
    
    close(fid)
    return rbm, train_params, split, gen, translated
end

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