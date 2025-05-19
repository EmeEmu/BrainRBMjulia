struct MomentsAggregate
    gen::Dict{String,Matrix}
    train::Dict{String,Matrix}
    valid::Dict{String,Matrix}
end

struct SimpleMomentsAggregate
    gen::Dict{String,Matrix}
    data::Dict{String,Matrix}
end

"""
    compute_moments(
        v::AbstractArray,
        h::AbstractArray;
        selected_indices::Union{Nothing, AbstractVector{<:Integer}}=nothing
    )::Dict{String, Matrix}

Compute <v>, <h>, <vh>, <vv> - <v><v>, and <hh> - <h><h> from visible and 
hidden activity matrices.

### Input

- `v`                -- visible activity matrix (visible x time)
- `h`                -- hidden activity matrix (hidden x time)
- `selected_indices` -- optional vector of visible unit indices to use 
                        for computing `<vv> - <v><v>`. If `nothing`,
                        all visible units are used.

### Output

- Dictionary with the following keys:
    - `"<v>"`               -- mean visible activity (visible x 1)
    - `"<h>"`               -- mean hidden activity (hidden x 1)
    - `"<vh>"`              -- cross-moment matrix (hidden x visible)
    - `"<vv> - <v><v>"`     -- covariance of visible units (length(selected_indices) x length(selected_indices))
    - `"<hh> - <h><h>"`     -- covariance of hidden units

### Notes

Visible activity is systematically converted to Float32.
"""
function compute_moments(
    v::AbstractArray,
    h::AbstractArray;
    selected_indices::Union{Nothing,AbstractVector{<:Integer}}=nothing
)::Dict{String,Matrix}

    lv = size(v, 2)
    lh = size(h, 2)
    v = Float32.(v)

    mu_v = mean(v, dims=2)
    mu_h = mean(h, dims=2)
    mu_vh = h * transpose(v) / lv

    if selected_indices !== nothing
        v_subset = v[selected_indices, :]
        cov_vv = cov(v_subset, dims=2, corrected=false)
    else
        cov_vv = cov(v, dims=2, corrected=false)
    end

    cov_hh = cov(h, dims=2, corrected=false)

    return Dict(
        "<v>" => mu_v,
        "<h>" => mu_h,
        "<vh>" => mu_vh,
        "<vv> - <v><v>" => cov_vv,
        "<hh> - <h><h>" => cov_hh,
    )
end


"""
    compute_all_moments(
        rbm,
        data::DatasetSplit,
        gen::GeneratedData;
        max_vv::Union{Nothing, Int}=nothing
    )::MomentsAggregate

Compute all relevant moments for generated, training, and validation datasets.

### Input

- `rbm`         -- the RBM model used to sample hidden states
- `data`        -- dataset split with `train` and `valid` matrices
- `gen`         -- generated data object with fields `v` and `h`
- `max_vv`      -- optional maximum number of visible units to use for 
                   `<vv> - <v><v>` computation in each dataset. If provided,
                   the same randomly selected visible indices are used across 
                   all three datasets.

### Output

- `MomentsAggregate` -- struct containing:
    - `gen`    -- moments of the generated data
    - `train`  -- moments of the training data
    - `valid`  -- moments of the validation data
"""
function compute_all_moments(
    rbm,
    data::DatasetSplit,
    gen::GeneratedData;
    max_vv::Union{Nothing,Int}=nothing
)::MomentsAggregate

    selected_indices = max_vv === nothing || max_vv >= size(data.train, 1) ?
                       nothing :
                       sort(randperm(size(data.train, 1))[1:max_vv])

    h_train = sample_h_from_v(rbm, data.train)
    h_valid = sample_h_from_v(rbm, data.valid)

    return MomentsAggregate(
        compute_moments(gen.v, gen.h; selected_indices=selected_indices),
        compute_moments(data.train, h_train; selected_indices=selected_indices),
        compute_moments(data.valid, h_valid; selected_indices=selected_indices),
    )
end


"""
    compute_all_moments(
        rbm,
        data::AbstractArray,
        gen::GeneratedData;
        max_vv::Union{Nothing, Int}=nothing
    )::SimpleMomentsAggregate

Compute moments for a single dataset and its corresponding generated data.

### Input

- `rbm`         -- the RBM model used to sample hidden states
- `data`        -- activity matrix (visible x time)
- `gen`         -- generated data object with fields `v` and `h`
- `max_vv`      -- optional maximum number of visible units to use for 
                   `<vv> - <v><v>` computation. If provided,
                   the same randomly selected visible indices are used
                   for both real and generated data.

### Output

- `SimpleMomentsAggregate` -- struct containing:
    - `gen`   -- moments of the generated data
    - `data`  -- moments of the input data
"""
function compute_all_moments(
    rbm,
    data::AbstractArray,
    gen::GeneratedData;
    max_vv::Union{Nothing,Int}=nothing
)::SimpleMomentsAggregate

    selected_indices = max_vv === nothing || max_vv >= size(data, 1) ?
                       nothing :
                       sort(randperm(size(data, 1))[1:max_vv])

    h_data = sample_h_from_v(rbm, data)

    return SimpleMomentsAggregate(
        compute_moments(gen.v, gen.h; selected_indices=selected_indices),
        compute_moments(data, h_data; selected_indices=selected_indices),
    )
end






function reconstruction_likelihood(rbm::Union{RBM,StandardizedRBM}, valid_spikes::AbstractArray, train_spikes::AbstractArray; eps=1.e-6)
    data = valid_spikes
    mu = reconstruct(rbm, data)
    x = ((eps .+ mu) .* data) .+ (1 .- mu .- eps) .* (1 .- data)
    likelihood = mean(log.(clamp.(x, 0, 1)), dims=2)

    indep = mean(train_spikes, dims=2)
    x = ((eps .+ indep) .* data) .+ (1 .- indep .- eps) .* (1 .- data)
    likelihood_indep = mean(log.(clamp.(x, 0, 1)), dims=2)

    normalised = (likelihood .- likelihood_indep) ./ (-likelihood_indep)
    return normalised
end;
