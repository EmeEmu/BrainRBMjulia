function swap_hidden_sign(rbm::StandardizedRBM{<:Any,<:xReLU})
    _rbm = deepcopy(cpu(rbm))
    for μ in 1:length(_rbm.hidden)
        if sum(_rbm.w[:,μ]) < 0
            _rbm.w[:,μ] .= -_rbm.w[:,μ]
            _rbm.hidden.θ[μ] = -_rbm.hidden.θ[μ]
            _rbm.hidden.ξ[μ] = -_rbm.hidden.ξ[μ]
            _rbm.offset_h[μ] = -_rbm.offset_h[μ]
        end
    end
    return _rbm
end