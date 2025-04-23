function dfsize()
    return Makie.current_default_theme().size[]
end

function span(x::AbstractArray)
    return abs(minimum(x) - maximum(x))
end

function span_dims(X::Matrix)
    D = size(X, 2)
    s = []
    for d in 1:1:D
        push!(s, span(X[:, d]))
    end
    return Tuple(s)
end

function centered_range(n::Int, step=1)
    a = collect(1:step:n*step)
    return a .- (1 + a[end]) / 2
end

function map_range(input, input_start, input_end, output_start, output_end)
    return output_start + ((output_end - output_start) / (input_end - input_start)) * (input - input_start)
end

function quantile_range(x::AbstractArray, q::Real=0.95)
    return maximum(abs.(quantile(vec(x), [1 - q, q])))
end

function vecmatTOarray(x)
    return permutedims(reshape(hcat(x...), size(x[1])..., length(x)), (3, 1, 2))
end
function vecvecTOarray(X)
    Y = [reshape(x, 1, size(x)...) for x in X]
    return permutedims(reshape(hcat(Y...), size(Y[1])..., length(Y)), (3, 1, 2))
end

function condenced_inds(n::Int, i::Int, j::Int)
    return binomial(n, 2) - binomial(n - i + 1, 2) + (j - i - 1) + 1
end
