function brainRBM(spikes::AbstractArray, M::Union{Int,Dims})
    N = size(spikes)[1]
    rbm = RBM(Binary((N,)), xReLU((M,)), randn(Float64, (N, M)) * 0.01)
    initialize!(rbm, spikes)
    return standardize(rbm)
end;

function stateRBM(rbm::StandardizedRBM, X::AbstractArray, S::Int)
    M = size(rbm.hidden, 1)
    B = Nbits_per_Nstates(S)    # number of bits in state layer
    S = Nstates_per_Nbits(B)    # number of states possible from layer of `B` binaries
    srbm = RBM(deepcopy(rbm.hidden), Binary((B,)), rand(Float64, (M, B)) * 0.01)

    #initialize!(srbm.hidden)
    #initialize_w!(srbm, X)
    #zerosum!(srbm)
    initialize!(srbm, X)

    return srbm, B, S
end



function translate(rbm::Union{RBM,StandardizedRBM}, spikes::AbstractArray)
    return mean_h_from_v(rbm, spikes)
end;

function reconstruct(rbm::Union{RBM,StandardizedRBM}, spikes::AbstractArray)
    return mean_v_from_h(rbm, mean_h_from_v(rbm, spikes))
end;

function build_training_h(
    rbm::Union{RBM,StandardizedRBM},
    spikes::AbstractArray;
    n::Int=50
)
    rh = sample_h_from_v(rbm, repeat(spikes, 1, 1, n))
    return reshape(rh, (size(rh, 1), size(rh, 2) * size(rh, 3)))
end
