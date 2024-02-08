struct BoxAround
  lims::Matrix{Float64}
  size::Vector{Int}
  origin::Vector{Float64}

  function BoxAround(coords::Matrix; padding::Float64=0.1)
    lims = vcat(maximum(coords, dims=1), minimum(coords, dims=1))
    scale = lims[1, :] - lims[2, :]
    lims .+= hcat(+scale .* padding, -scale .* padding)'
    new(lims, round.(Int, lims[1, :] - lims[2, :]), lims[2, :])
  end
  # function BoxAround(coords::Matrix; padding::Int=100)
  #   lims = vcat(maximum(coords, dims=1), minimum(coords, dims=1))
  #   lims .+= hcat(+padding, -padding)'
  #   new(lims, round.(Int, lims[1, :] - lims[2, :]), lims[2, :])
  # end
end

function ball_mask(R::Int)
  bs = 2 * R + 1
  ballnan = fill!(Array{Float64}(undef, bs, bs, bs), NaN)
  ballint = zeros(Int, bs, bs, bs)
  c = (bs ÷ 2) + 1
  for i in 1:bs
    for j in 1:bs
      for k in 1:bs
        d2 = (i - c)^2 + (j - c)^2 + (k - c)^2
        if d2 < (R + 0.1)^2
          ballnan[i, j, k] = 1
          ballint[i, j, k] = 1
        end
      end
    end
  end
  return ballnan, ballint
end

function build_map(coords::Matrix{Float32}, weights::Vector{Float32}, box::BoxAround; R=4)
  ballnan, ballint = ball_mask(R)
  cint = round.(Int, coords .- box.origin')

  mapint = zeros(Int, box.size...)
  for c in 1:size(cint, 1)
    i, j, k = cint[c, :]
    mapint[i-R:i+R, j-R:j+R, k-R:k+R] += ballint
  end


  map = Array{Float32}(undef, box.size...)
  map[mapint.==0] .= NaN
  for c in 1:size(cint, 1)
    i, j, k = cint[c, :]
    map[i-R:i+R, j-R:j+R, k-R:k+R] .+= ballint .* weights[c]
  end

  return map ./= mapint
end

function build_map(coords::Matrix{Float32}, weights::Matrix{Float32}, box::BoxAround; R=4)
  map = Array{Float32}(undef, size(weights, 2), box.size...)
  @showprogress for i in 1:size(weights, 2)
    map[i, :, :, :] = build_map(coords, weights[:, i], box; R)
  end
  return map
end


function gaussian(x, y, z, σ=1)
  return (1 / (2 * π * σ^2)) * exp(-(x^2 + y^2 + z^2) / (2 * σ)^2)
end
function gaussian_kernel(; k=100, σ=5)
  c = k ÷ 2
  K = [gaussian(i - c, j - c, k - c, σ) for i = 1:k, j = 1:k, k = 1:k]
  return K ./ sum(K)
end

function smooth(map::Array{Float32,3}, kernel::Array{Float32,3})
  K = size(kernel, 1) ÷ 2

  S = Array{Float32}(undef, size(map))
  S .= NaN32

  ini = K + 1
  fin = size(map) .- K
  @floop for i in ini:fin[1], j in ini:fin[2], k in ini:fin[3]
    @views S[i, j, k] = nanmean(map[i-K:i+K, j-K:j+K, k-K:k+K])
  end

  return S
end

function smooth(map::Array{Float32,3}, σ::Real)
  k = trunc(Int, 4 * σ)
  k = (k ÷ 2) * 2 + 1
  kernel = Float32.(gaussian_kernel(; k, σ))
  return smooth(map, kernel)
end



function KL_divergence(p, q)
  return sum(p .* log2.(p ./ q))
end

function JS_divergence(p, q)
  m = (p .+ q) ./ 2
  kl_pm = KL_divergence(p, m)
  kl_qm = KL_divergence(q, m)
  return 0.5 * kl_pm + 0.5 * kl_qm
end

function JS_distance(p, q)
  return sqrt(JS_divergence(p, q))
end

function JS_distance(ps::Array{Float32,4})
  n = size(ps, 1)
  JSD = Matrix{Float64}(undef, (n, n))
  @floop for i in 1:n
    for j in i+1:n
      JSD[i, j] = JSD[j, i] = JS_distance(ps[i, :, :, :], ps[j, :, :, :])
    end
    JSD[i, i] = 0
  end
  return JSD
end
