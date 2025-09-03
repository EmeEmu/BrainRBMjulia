"""
    coupling_approx(rbm::Union{RBM,StandardizedRBM}, v::AbstractArray)

Approximate the effective coupling matrix between visible units.
The approximation averages the conditional variance of hidden units
given the visible configuration `v` and projects it through the weights
of `rbm`.

Returns a symmetric matrix `J` with zeros on its diagonal.
"""
function coupling_approx(rbm::Union{RBM,StandardizedRBM}, v::AbstractArray)
  I = inputs_h_from_v(rbm, v)
  cond_var = var_from_inputs(rbm.hidden, I)
  mean_var = mean(cond_var, dims=2)
  J = rbm.w * (rbm.w .* reshape(mean_var, 1, :))'
  J[diagind(J)] .= 0
  return J
end
