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

function int_coords(coords::AbstractMatrix, box::BoxAround)
  return round.(Int, coords .- box.origin')
end

function create_map(coords::AbstractMatrix, weights::AbstractVector, box::BoxAround; R::Int=4, verbose::Bool=true)
  _, ballint = ball_mask(R)
  cint = int_coords(coords, box)

  mapint = zeros(Int, box.size...)
  map = zeros(Float32, box.size...)
  progBar = Progress(size(cint, 1), dt=0.1, desc="Creating Map: ", showspeed=true, enabled=verbose)
  for c in 1:size(cint, 1)
    i, j, k = cint[c, :]
    mapint[i-R:i+R, j-R:j+R, k-R:k+R] += ballint
    map[i-R:i+R, j-R:j+R, k-R:k+R] .+= ballint .* weights[c]
    next!(progBar)
  end

  return map ./= mapint
end
function create_map(coords::AbstractMatrix, weights::AbstractMatrix, box::BoxAround; R=4, verbose::Bool=true)
  maps = Array{Float32}(undef, size(weights, 2), box.size...)
  progBar = Progress(size(weights, 2), dt=0.1, desc="Creating Maps: ", showspeed=true, enabled=verbose)
  for i in 1:size(weights, 2)
    maps[i, :, :, :] = create_map(coords, weights[:, i], box; R, verbose=false)
    next!(progBar)
  end
  return maps
end

function map_finite!(map::AbstractArray; val::Real=0)
  return replace!(map, Inf => val, -Inf => val, NaN => val)
end

function gaussian3D(box::BoxAround, σ::Real=1)
  x0, y0, z0 = box.size .÷ 2
  gauss(x, y, z) = exp(-((x - x0)^2 + (y - y0)^2 + (z - z0)^2) / (2 * σ^2))
  xs, ys, zs = [1:box.size[i] for i in 1:3]
  G = [gauss(x, y, z) for x in xs, y in ys, z in zs]
  return G
end

function interpolate_map(coords::AbstractMatrix, map::AbstractArray{T,3}, box::BoxAround) where {T<:AbstractFloat}
  cint = int_coords(coords, box)
  return [map[cint[i, :]...] for i in 1:size(cint, 1)]
end
function interpolate_map(coords::AbstractMatrix, maps::AbstractArray{T,4}, box::BoxAround; verbose=true) where {T<:AbstractFloat}
  w = Array{typeof(maps[1])}(undef, size(coords, 1), size(maps, 1))
  progBar = Progress(size(maps, 1), dt=0.1, desc="Interpolating Maps: ", showspeed=true, enabled=verbose)
  for i in 1:size(maps, 1)
    w[:, i] .= interpolate_map(coords, maps[i, :, :, :], box)
    next!(progBar)
  end
  return w
end

function smooth_map(map::AbstractArray{T,3}, box::BoxAround, σ::Real) where {T<:AbstractFloat}
  K = gaussian3D(box, σ)
  K ./= sum(K)
  sK = fftshift(K)
  F = plan_rfft(K)
  fK = F * sK
  fmap = F * map
  return irfft(fK .* fmap, size(K, 1))
end
function smooth_map(maps::AbstractArray{T,4}, box::BoxAround, σ::Real; verbose::Bool=true) where {T<:AbstractFloat}
  K = gaussian3D(box, σ)
  K ./= sum(K)
  sK = fftshift(K)
  F = plan_rfft(K)
  fK = F * sK
  iF = plan_irfft(fK, size(K, 1))
  M = copy(maps)
  progBar = Progress(size(maps, 1), dt=0.1, desc="Smoothing Maps: ", showspeed=true, enabled=verbose)
  for i in 1:size(maps, 1)
    fmap = F * maps[i, :, :, :]
    M[i, :, :, :] .= iF * (fK .* fmap)
    next!(progBar)
  end
  return M
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
