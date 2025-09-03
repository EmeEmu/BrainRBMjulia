"""
    dfsize()

Return the default figure size from the current Makie theme.

This helper is useful when you want to generate figures that match
the dimensions of other plots created with the default theme.
"""
function dfsize()
    return Makie.current_default_theme().size[]
end

"""
    span(x)

Compute the absolute range of values in `x`.

`span` returns the difference between the minimum and maximum values of
an array and is commonly used for normalising data ranges.
"""
function span(x::AbstractArray)
    return abs(minimum(x) - maximum(x))
end

"""
    span_dims(X)

Return the span of each column of matrix `X`.

The result is a tuple where the `i`-th element corresponds to the range
of values along the `i`-th dimension (column) of `X`.
"""
function span_dims(X::Matrix)
    D = size(X, 2)
    s = []
    for d in 1:1:D
        push!(s, span(X[:, d]))
    end
    return Tuple(s)
end

"""
    centered_range(n, step=1)

Generate a symmetric range centered around zero.

The returned vector contains `n` evenly spaced numbers separated by `step`
such that the mean of the sequence is zero.
"""
function centered_range(n::Int, step=1)
    a = collect(1:step:n*step)
    return a .- (1 + a[end]) / 2
end

"""
    map_range(input, input_start, input_end, output_start, output_end)

Linearly map `input` from one range to another.

The value `input` is assumed to lie within [`input_start`, `input_end`]
and is mapped to the corresponding position in
[`output_start`, `output_end`].
"""
function map_range(input, input_start, input_end, output_start, output_end)
    return output_start + ((output_end - output_start) / (input_end - input_start)) * (input - input_start)
end

"""
    quantile_range(x, q=0.95)

Return a symmetric range based on the `q` quantile of `x`.

The maximum absolute value of the lower and upper `q`-quantiles is
returned. This is useful for setting colour ranges in plots.
"""
function quantile_range(x::AbstractArray, q::Real=0.95)
    return maximum(abs.(quantile(vec(x), [1 - q, q])))
end

"""
    vecmatTOarray(x)

Stack a vector of matrices into a 3‑D array.

Each matrix in `x` must have identical dimensions. The result has size
`(length(x), size(x[1])...)` with matrices stored along the first
dimension.
"""
function vecmatTOarray(x)
    return permutedims(reshape(hcat(x...), size(x[1])..., length(x)), (3, 1, 2))
end

"""
    vecvecTOarray(X)

Convert a vector of vectors to a 3‑D array.

Vectors are treated as 1×N matrices before stacking, mirroring the layout
produced by [`vecmatTOarray`](@ref).
"""
function vecvecTOarray(X)
    Y = [reshape(x, 1, size(x)...) for x in X]
    return permutedims(reshape(hcat(Y...), size(Y[1])..., length(Y)), (3, 1, 2))
end

"""
    condenced_inds(n, i, j)

Return the linear index of the upper‑triangular pair `(i, j)`.

The matrix is assumed to be condensed such that only `i < j` pairs are
stored. This helper is typically used when arranging pairwise plots.
"""
function condenced_inds(n::Int, i::Int, j::Int)
    return binomial(n, 2) - binomial(n - i + 1, 2) + (j - i - 1) + 1
end
