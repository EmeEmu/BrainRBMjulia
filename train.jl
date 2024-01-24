"""
    training_wrapper(
        rbm::StandardizedRBM, spikes::AbstractArray;
        iters::Int=20000,batchsize::Int=256,steps::Int=50,
        lr_start::Number=5f-4, lr_stop::Number=1f-5, decay_from::Number=0.25,
        l2l1::Number=0.001, l1::Number=0,
        ϵv::Number=1f-1, ϵh::Number=0f0, damping::Number=1f-1,
        record_ps::Bool=true, verbose::Bool=true
        )::Tuple{
              ValueHistories.MVHistory{ValueHistories.History},
              Dict{String, Real}
                }

Training a StandardizedRBM from neuron activity. A wrapper of the pcd! method from
RestrictedBoltzmannMachines.jl (see https://github.com/cossio/RestrictedBoltzmannMachines.jl/blob/master/src/train/pcd.jl).
Adds geometricaly decaying learning rate, 


### Input

- `rbm`         -- StandardizedRBM instance; 
                    rbm initiated and pushed to the GPU
- `spikes`      -- neuron activity matrix; 
                    (neurons x time) matrix pushed to the GPU
- `iters`       -- (optional, default: `20000`) number of gradient updates; 
                    parsed to pcd!
- `batchsize`   -- (optional, default: `256`) batch size for gradient updates; 
                    parsed to pcd!
- `steps`       -- (optional, default: `50`) number of Markov Chain steps to 
                    update the fantasy chain; 
                    parsed to pcd!
- `lr_start`    -- (optional, default: `5f-4`) starting learning rate;
- `lr_strop`    -- (optional, default: `1f-5`) final learning rate;
- `decay_from`  -- (optional, default: `0.25`) fraction of the training from 
                    which to start lerning rate decay;
- `l2l1`        -- (optional, default: `1f-3`) L2L1 regularization of weights;
                    parsed to pcd!
- `l1`          -- (optional, default: `0`) L1 regularization of weights;
                    parsed to pcd!
- `ϵv`          -- (optional, default: `1f-1`) pseudocount for variance 
                    estimation of visible units;
                    parsed to pcd!
- `ϵh`          -- (optional, default: `0f0`) pseudocount for variance 
                    estimation of hidden units;
                    parsed to pcd!
- `damping`     -- (optional, default: `1f-1`) damping ??;
                    parsed to pcd!
- `record_ps`   -- (optional, default: `true`) record pseudolikelihood during 
                    training; 
                    cannot be used for certain potential (eg. Gaussian)
- `verbose`     -- (optional, default: `true`) show progress bar;

### Output

- `history`     -- value history of learning rate 
                    (+ pseudolikelihood if `record_ps=true`);
- `params`      -- dictionary of parameters

### Notes

Using this function is not a guarentee of convergence. Proper convergence
should be assessed systematicaly.
"""
function training_wrapper(
  rbm::StandardizedRBM,
  spikes::AbstractArray; iters::Int=20000, # 50000
  batchsize::Int=256, # 256
  steps::Int=50, # 50
  lr_start::Number=5.0f-4, # 1f-4
  lr_stop::Number=1.0f-5, # 1f-5
  decay_from::Number=0.25, # 0.25
  l2l1::Number=0.001, # 0.001
  l1::Number=0, # 0
  ϵv::Number=1.0f-1, # 1f-1
  ϵh::Number=0.0f0, # 0f0
  damping::Number=1.0f-1,
  record_ps::Bool=true, # true
  verbose::Bool=true # true
)

  decay_g = (lr_stop / lr_start)^(1 / (iters * (1 - decay_from)))
  history = MVHistory()
  progBar = Progress(iters, dt=0.1, desc="Training: ", showspeed=true, enabled=verbose)

  function callback(; rbm, optim, state, ps, iter, vm, vd, ∂)
    # learning rate section
    lr = state.w.rule.eta
    if iter > decay_from * iters
      adjust!(state, lr * decay_g)
    end
    @trace history iter lr


    # progress bar section
    next!(progBar)

    # pseudolikelihood section

    if iszero(iter % 200) & record_ps
      lpl = mean(log_pseudolikelihood(rbm, spikes))
      @trace history iter lpl
    end


  end

  optim = Adam(lr_start, (9.0f-1, 999.0f-3), 1.0f-6) # (0f0, 999f-3), 1f-6
  n = size(rbm.visible)[1]
  vm = sample_from_inputs(rbm.visible, gpu(zeros(n, batchsize)))
  state, ps = pcd!(
    rbm, spikes;
    optim,
    steps=steps,
    batchsize,
    iters=iters,
    vm,
    l2l1_weights=l2l1,
    l1_weights=l1,
    ϵv=ϵv, # 1f-1
    ϵh=ϵh, # 0f0
    damping=damping,
    callback
  )

  return history, Dict([
    ("iters", iters),
    ("batchsize", batchsize),
    ("steps", steps),
    ("lr_start", lr_start),
    ("lr_stop", lr_stop),
    ("decay_from", decay_from),
    ("l2l1", l2l1),
    ("l1", l1),
    ("ϵv", ϵv),
    ("ϵh", ϵh),
    ("damping", damping),
  ])
end
