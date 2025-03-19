function state_sampling(
    rbm::Union{RBM,StandardizedRBM},
    srbm::Union{RBM,StandardizedRBM},
    spikes::AbstractArray;
    n::Int=50
)
    rh = sample_h_from_v(rbm, repeat(spikes, 1, 1, n))
    rb = sample_h_from_v(srbm, rh)
    rs = decode_binary(rb)
    return rs
end
function state_sampling(
    srbm::Union{RBM,StandardizedRBM},
    hs::AbstractArray;
    n::Int=50
)
    rb = sample_h_from_v(srbm, repeat(hs, 1, 1, n))
    rs = decode_binary(rb)
    return rs
end

function state_distrib(rs::Vector{Int64}, S::Int)
    b = fit(Histogram, rs, 0:1:S).weights
    p = b ./ sum(b)
    return p
end
function state_distrib(rs::Matrix{Int64}, S::Int)
    b = hcat([fit(Histogram, rs[i, :], 0:1:S).weights for i in 1:size(rs, 1)]...)
    p = b ./ sum(b, dims=1)
    return p
end
function state_distrib(
    rbm::Union{RBM,StandardizedRBM},
    srbm::Union{RBM,StandardizedRBM},
    spikes::AbstractArray;
    n::Int=50
)
    rs = state_sampling(rbm, srbm, spikes; n)
    return state_distrib(rs, Nstates_per_Nbits(length(srbm.hidden.θ)))
end
function state_distrib(
    srbm::Union{RBM,StandardizedRBM},
    hs::AbstractArray;
    n::Int=50
)
    rs = state_sampling(srbm, hs; n)
    return state_distrib(rs, Nstates_per_Nbits(length(srbm.hidden.θ)))
end

function state_max(ps::Matrix{Float64})
    states = [c[1] for c in argmax(ps, dims=1)[1, :]]
    probs = [ps[states[t], t] for t in 1:size(ps, 2)]
    return states .- 1, probs
end

function state_proba(ps::Matrix{Float64})
    return mean(ps, dims=2)[:, 1]
end

function state_transition(ps::Matrix{Float64})
    T = sum([ps[:, t] * ps[:, t+1]' for t in 1:size(ps, 2)-1])
    T ./= sum(T, dims=2)
    return T
end


function mean_v_by_s(
    rbm::Union{RBM,StandardizedRBM},
    srbm::Union{RBM,StandardizedRBM},
    spikes::AbstractArray;
    n::Int=50,
    steps::Int=1
)
    vs = repeat(spikes, 1, 1, n)
    vs = reshape(vs, (size(vs, 1), size(vs, 2) * size(vs, 3)))
    vs = sample_v_from_v(rbm, vs; steps)

    hs = sample_h_from_v(rbm, vs)

    bs = sample_h_from_v(srbm, hs)
    ss = decode_binary(bs)
    S = Nstates_per_Nbits(length(srbm.hidden))

    means = mapreduce(
        permutedims,
        vcat,
        [mean(vs[:, findall(ss .== s)], dims=2)[:, 1] for s in 0:S-1]
    )
    stds = mapreduce(
        permutedims,
        vcat,
        [std(vs[:, findall(ss .== s)], dims=2)[:, 1] for s in 0:S-1]
    )

    return means, stds
end
