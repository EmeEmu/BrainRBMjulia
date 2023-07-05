function training_wrapper(
        rbm::StandardizedRBM, 
        spikes::AbstractArray;
        
        iters::Int=20000, # 50000
        batchsize::Int=256, # 256
        steps::Int=50, # 50
        lr_start::Number=5f-4, # 1f-4
        lr_stop::Number=1f-5, # 1f-5
        decay_from::Number=0.25, # 0.25
        l2l1::Number=0.001, # 0.001
        l1::Number=0, # 0
        record_ps::Bool=true, # true
    )
    
    decay_g = (lr_stop/lr_start)^(1/(iters*(1-decay_from)))
    history = MVHistory()
    progBar = Progress(iters, dt=0.1, desc="Training: ", showspeed=true);
    
    function callback(; rbm, optim, state, ps, iter, vm, vd, ∂)
        # learning rate section
        lr = state.w.rule.eta
        if iter > decay_from*iters
            adjust!(state, lr*decay_g)
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
    
    optim = Adam(lr_start, (9f-1, 999f-3), 1f-6) # (0f0, 999f-3), 1f-6
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
        ϵv=1f-1, # 1f-1
        ϵh=0f0, # 0f0
        damping=1f-1,
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
            ("ϵv", 1f-1),
            ("ϵh", 0f0),
            ("damping", 1f-1),
        ])
end