"""
    zscore(A)

Normalize each row of `A` to zero mean and unit variance.

Returns an array where every row has been z-scored independently.
"""
function zscore(A)
  μs = mean(A, dims=2)
  σs = std(A, dims=2)
  return (A .- μs) ./ σs
end

"""
    quantile_2d(x::Matrix, p)

Compute the `p`-quantile along the second dimension of matrix `x`.

The result is a row vector containing the quantile of each row of `x`.
"""
function quantile_2d(x::Matrix, p)
  a = [quantile(x[i, :], p) for i in 1:size(x, 1)]
  return permutedims(hcat(a...))
end
