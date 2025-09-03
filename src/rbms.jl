"""
    BrainRBM(spikes, M)

Construct a standardized [`RBM`](@ref) with binary visible units and `xReLU`
hidden units from the training data `spikes`.  Weights are drawn from a normal
distribution with a small standard deviation and the network is initialized and
standardized using the provided data.

# Arguments
- `spikes::AbstractArray`: training data whose first dimension corresponds to
  the number of visible units.
- `M::Union{Int,Dims}`: number of hidden units or their dimensions.

# Returns
A [`StandardizedRBM`](@ref) ready for training.
"""
function BrainRBM(spikes::AbstractArray, M::Union{Int,Dims})
    N = size(spikes)[1]
    rbm = RBM(Binary((N,)), xReLU((M,)), randn(Float64, (N, M)) * 0.01)
    initialize!(rbm, spikes)
    return standardize(rbm)
end;

"""
    StateRBM(rbm, X, S)

Create an RBM with an additional binary state layer capable of representing
`S` distinct states.  The returned model shares the hidden layer of `rbm` and
adds `B` binary units, where `B = Nbits_per_Nstates(S)`.

# Arguments
- `rbm::StandardizedRBM`: base model whose hidden layer is used as the visible
  layer of the new RBM.
- `X::AbstractArray`: data used to initialize the state RBM.
- `S::Int`: desired number of states to encode.

# Returns
A tuple `(srbm, B, S)` containing the new state RBM, the number of bits `B`
in the state layer, and the actual number of representable states `S`.
"""
function StateRBM(rbm::StandardizedRBM, X::AbstractArray, S::Int)
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



"""
    translate(rbm, spikes)

Compute the mean hidden activations produced by presenting `spikes` to `rbm`.
"""
function translate(rbm::Union{RBM,StandardizedRBM}, spikes::AbstractArray)
    return mean_h_from_v(rbm, spikes)
end;

"""
    reconstruct(rbm, spikes)

Encode `spikes` into the hidden layer and decode them back to reconstruct the
visible activity.
"""
function reconstruct(rbm::Union{RBM,StandardizedRBM}, spikes::AbstractArray)
    return mean_v_from_h(rbm, mean_h_from_v(rbm, spikes))
end;

"""
    build_training_h(rbm, spikes; n=50)

Generate a hidden-layer training set by sampling the hidden units `n` times for
each column of `spikes` and flattening the result into a two-dimensional array.

# Arguments
- `rbm::Union{RBM,StandardizedRBM}`: model from which to sample.
- `spikes::AbstractArray`: visible data used to drive the RBM.
- `n`: number of hidden samples per input example.

# Returns
An array of size `(hidden unis, time*n)` containing the sampled
hidden activations.
"""
function build_training_h(
    rbm::Union{RBM,StandardizedRBM},
    spikes::AbstractArray;
    n::Int=50
)
    rh = sample_h_from_v(rbm, repeat(spikes, 1, 1, n))
    return reshape(rh, (size(rh, 1), size(rh, 2) * size(rh, 3)))
end
