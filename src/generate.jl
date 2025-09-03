"""
    GeneratedData

Container for samples and metadata produced by [`gen_data`](@ref).

### Fields

- `v`         – visible samples (`n_visible × nsamples`)
- `h`         – hidden samples (`n_hidden × nsamples`)
- `thermal`   – mean free energy during thermalization
- `nsamples`  – number of samples drawn after thermalization
- `nthermal`  – number of thermalization iterations
- `nstep`     – Gibbs steps between successive samples
"""
struct GeneratedData
    v::Union{Matrix, BitMatrix}
    h::Union{Matrix, BitMatrix}
    thermal::Vector
    nsamples::Int
    nthermal::Int
    nstep::Int
end

"""
    gen_data(rbm::Union{RBM, StandardizedRBM};
             nsamples::Int=5000,
             nthermal::Int=500,
             nstep::Int=100,
             init::String="prior",
             verbose::Bool=true)

Generate visible and hidden samples from `rbm` using block Gibbs sampling.
The returned [`GeneratedData`](@ref) stores the sampled states and metadata
about the procedure.
"""
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
