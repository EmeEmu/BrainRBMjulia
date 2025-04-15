struct GeneratedData
    v::Union{Matrix, BitMatrix}
    h::Union{Matrix, BitMatrix}
    thermal::Vector
    nsamples::Int
    nthermal::Int
    nstep::Int
end

function gen_data(rbm::Union{RBM, StandardizedRBM};nsamples::Int=5000, nthermal::Int=500,nstep::Int=100, init::String="prior", verbose::Bool=true)
    if init=="random"
        if isa(rbm.visible.par, CUDA.CuArray)
            sampled_v = gpu(rand(typeof(rbm.visible.par[1]), size(rbm.visible, 1), nsamples));
        else
            sampled_v = rand(typeof(rbm.visible.par[1]), size(rbm.visible, 1), nsamples);
        end
    elseif init=="prior"
        if isa(rbm.visible.par, CUDA.CuArray)
            sampled_v = sample_from_inputs(rbm.visible, gpu(zeros(size(rbm.visible, 1), nsamples)))
        else
            sampled_v = sample_from_inputs(rbm.visible, zeros(size(rbm.visible, 1), nsamples))
        end
    end
    sampled_f = zeros(nthermal)
    if verbose
        @progress name = "Thermalization" for t in 1:nthermal
            sampled_v .= sample_v_from_v(rbm, sampled_v; steps=nstep)
            sampled_f[t] = mean(free_energy(rbm, sampled_v))
        end
    else
        for t in 1:nthermal
            sampled_v .= sample_v_from_v(rbm, sampled_v; steps=nstep)
            sampled_f[t] = mean(free_energy(rbm, sampled_v))
        end
    end

   
    if isa(rbm.visible.par, CUDA.CuArray)
        return GeneratedData(
            cpu(sampled_v), 
            cpu(sample_h_from_v(rbm, sampled_v)),
            sampled_f,
            nsamples,
            nthermal,
            nstep,
        )
    else
        return GeneratedData(
            sampled_v, 
            sample_h_from_v(rbm, sampled_v),
            sampled_f,
            nsamples,
            nthermal,
            nstep,
        )
    end
end
