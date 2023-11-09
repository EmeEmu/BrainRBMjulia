function coupling_approx(rbm::Union{RBM,StandardizedRBM}, v::AbstractArray)
  I = inputs_h_from_v(rbm, v)
  cond_var = var_from_inputs(rbm.hidden, I)
  mean_var = mean(cond_var, dims=2)
  J = rbm.w' * (rbm.w .* reshape(mean_var, 1, :))
  J[diagind(J)] .= 0
  return J
end
