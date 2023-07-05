function BrainRBM(spikes::AbstractArray, M::Union{Int,Dims})
    N = size(spikes)[1]
    rbm = RBM(Binary((N,)), xReLU((M,)), randn(Float64, (N,M))*0.01)
    initialize!(rbm, spikes)
    return standardize(rbm)
end;




function translate(rbm::Union{RBM, StandardizedRBM}, spikes::AbstractArray)
    return mean_h_from_v(rbm, spikes);
end;

function reconstruct(rbm::Union{RBM, StandardizedRBM}, spikes::AbstractArray)
    return mean_v_from_h(rbm, mean_h_from_v(rbm, spikes));
end;