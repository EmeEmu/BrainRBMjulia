function zscore(A)
  μs = mean(A, dims=2)
  σs = std(A, dims=2)
  return (A .- μs) ./ σs
end

function quantile_2d(x::Matrix, p)
  a = [quantile(x[i, :], p) for i in 1:size(x, 1)]
  return permutedims(hcat(a...))
end
