function compute_moments(v::AbstractArray, h::AbstractArray)
    lv = size(v)[2];
    lh = size(h)[2];
    v = Float32.(v);
    mu_v = mean(v, dims=2);
    mu_h = mean(h, dims=2);
    mu_vh = h*transpose(v) / lv;
    cov_vv = cov(v, dims=2, corrected=false);#v*transpose(v) / lv - mu_v*transpose(mu_v);
    cov_hh = cov(h, dims=2, corrected=false);#h*transpose(h) / lh - mu_h*transpose(mu_h);
    return Dict([
            ("<v>", mu_v),
            ("<h>", mu_h),
            ("<vh>", mu_vh),
            ("<vv> - <v><v>", cov_vv),
            ("<hh> - <h><h>", cov_hh),
            ])
end

struct MomentsAggregate
    gen::Dict{String, Matrix}
    train::Dict{String, Matrix}
    valid::Dict{String, Matrix}
end

function compute_all_moments(rbm, data::DatasetSplit, gen::GeneratedData)
    h_train = sample_h_from_v(rbm, data.train)
    h_valid = sample_h_from_v(rbm, data.valid)
    return MomentsAggregate(
        compute_moments(gen.v, gen.h),
        compute_moments(data.train, h_train),
        compute_moments(data.valid, h_valid),
    )
end


function reconstruction_likelihood(rbm::Union{RBM, StandardizedRBM}, valid_spikes::AbstractArray, train_spikes::AbstractArray ;eps=1.e-6)
    data = valid_spikes;
    mu = reconstruct(rbm, data);
    x = ((eps .+ mu) .* data) .+ (1 .- mu .- eps) .* ( 1 .- data)
    likelihood = mean(log.( clamp.(x, 0, 1) ), dims=2) ;

    indep = mean(train_spikes, dims=2);
    x = ((eps .+ indep) .* data) .+ (1 .- indep .- eps) .* ( 1 .- data)
    likelihood_indep = mean(log.( clamp.(x, 0, 1) ), dims=2) ;

    normalised = (likelihood.-likelihood_indep) ./ (-likelihood_indep);
    return normalised
end;