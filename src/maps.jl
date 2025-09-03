"""
    BoxAround

Axis-aligned bounding box enclosing a set of coordinates. Stores limits,
grid size and origin, and provides constructors from raw coordinates or
explicit parameters.
"""
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

  function BoxAround(lims::Matrix{Float64}, size::Vector{Int}, origin::Vector{Float64})
    new(lims, size, origin)
  end
end

"""
    ball_mask(R::Int)

Generate 3‑D spherical masks of radius `R`. Returns a pair of arrays: a
`Float64` mask with `NaN` outside the ball and an integer mask of ones and
zeros.
"""
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

"""
    int_coords(coords, box::BoxAround)

Convert continuous coordinates to integer indices within `box`.
"""
function int_coords(coords::AbstractMatrix, box::BoxAround)
  return round.(Int, coords .- box.origin')
end

"""
    create_map(coords, weights, box; R=4, verbose=true)

Rasterise weighted points at `coords` into a 3‑D grid defined by `box` using a
spherical kernel of radius `R`. If `weights` is a matrix, a stack of maps is
returned.
"""
function create_map(coords::AbstractMatrix, weights::AbstractVector, box::BoxAround; R::Int=4, verbose::Bool=true)
  _, ballint = ball_mask(R)
  cint = int_coords(coords, box)

  mapint = zeros(Int, box.size...)
  map = zeros(Float32, box.size...)
  # progBar = Progress(size(cint, 1), dt=0.1, desc="Creating Map: ", showspeed=true, enabled=verbose)
  for c in 1:size(cint, 1)
    i, j, k = cint[c, :]
    mapint[i-R:i+R, j-R:j+R, k-R:k+R] += ballint
    map[i-R:i+R, j-R:j+R, k-R:k+R] .+= ballint .* weights[c]
    # next!(progBar)
  end

  return map ./= mapint
end
function create_map(coords::AbstractMatrix, weights::AbstractMatrix, box::BoxAround; R=4, verbose::Bool=true)
  maps = Array{Float32}(undef, size(weights, 2), box.size...)
  # progBar = Progress(size(weights, 2), dt=0.1, desc="Creating Maps: ", showspeed=true, enabled=verbose)
  for i in 1:size(weights, 2)
    maps[i, :, :, :] = create_map(coords, weights[:, i], box; R, verbose=false)
    # next!(progBar)
  end
  return maps
end

"""
    map_finite(map; val=0)

Replace `Inf`, `-Inf` and `NaN` values in `map` with `val`.
"""
function map_finite(map::AbstractArray; val::Real=0)
  return replace(map, Inf => val, -Inf => val, NaN => val)
end

"""
    map_finite!(map; val=0)

In-place version of [`map_finite`](@ref).
"""
function map_finite!(map::AbstractArray; val::Real=0)
  return replace!(map, Inf => val, -Inf => val, NaN => val)
end


# FFT smoothing interpolation ________________________________________________
"""
    gaussian3D(box::BoxAround, σ::Real=1)

Generate a 3‑D Gaussian kernel matching the dimensions of `box`.
"""
function gaussian3D(box::BoxAround, σ::Real=1)
  x0, y0, z0 = box.size .÷ 2
  gauss(x, y, z) = exp(-((x - x0)^2 + (y - y0)^2 + (z - z0)^2) / (2 * σ^2))
  xs, ys, zs = [1:box.size[i] for i in 1:3]
  G = [gauss(x, y, z) for x in xs, y in ys, z in zs]
  return G
end

"""
    interpolate_map(coords, map, box)
    interpolate_map(coords, maps, box; verbose=true)

Sample values from a single map or a stack of maps at given `coords`.
"""
function interpolate_map(coords::AbstractMatrix, map::AbstractArray{T,3}, box::BoxAround) where {T<:AbstractFloat}
  cint = int_coords(coords, box)
  return [map[cint[i, :]...] for i in 1:size(cint, 1)]
end
function interpolate_map(coords::AbstractMatrix, maps::AbstractArray{T,4}, box::BoxAround; verbose=true) where {T<:AbstractFloat}
  w = Array{typeof(maps[1])}(undef, size(coords, 1), size(maps, 1))
  # progBar = Progress(size(maps, 1), dt=0.1, desc="Interpolating Maps: ", showspeed=true, enabled=verbose)
  for i in 1:size(maps, 1)
    w[:, i] .= interpolate_map(coords, maps[i, :, :, :], box)
    # next!(progBar)
  end
  return w
end

"""
    smooth_map(map, box, σ)
    smooth_map(maps, box, σ; verbose=true)

Smooth map(s) with a Gaussian kernel using FFT convolution.
"""
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
# FFT smoothing interpolation ________________________________________________


# ball gaussian interpolation ________________________________________________
"""
    nan_mean(x, w)

Weighted sum of `x` ignoring non-finite values, using weights `w`.
"""
function nan_mean(x::AbstractArray, w::AbstractArray)
  y = x .* w
  idx = isfinite.(y)
  return sum(y[idx])
end

"""
    region(map, xyz, R)

Extract cubic region of radius `R` around `xyz` from `map`.
"""
function region(map::AbstractArray{T,3}, xyz::Vector{Int}, R::Int) where {T<:AbstractFloat}
  x, y, z = xyz
  return map[x-R:x+R, y-R:y+R, z-R:z+R]
end

"""
    regionmean(map, xyz, K, R)
    regionmean(map, xyzs, K, R)
    regionmean(maps, xyzs, K, R; verbose=true)

Average regions of `map` or `maps` weighted by kernel `K`.
"""
function regionmean(map::AbstractArray{T,3}, xyz::Vector{Int}, K::AbstractArray{T,3}, R::Int) where {T<:AbstractFloat}
  return nan_mean(region(map, xyz, R), K)
end
function regionmean(map::AbstractArray{T,3}, xyz::Matrix{Int}, K::AbstractArray{T,3}, R::Int) where {T<:AbstractFloat}
  return [regionmean(map, xyz[i, :], K, R) for i in 1:size(xyz, 1)]
end
function regionmean(maps::AbstractArray{T,4}, xyz::Matrix{Int}, K::AbstractArray{T,3}, R::Int; verbose=true) where {T<:AbstractFloat}
  w = Array{typeof(maps[1])}(undef, size(xyz, 1), size(maps, 1))
  # progBar = Progress(size(maps, 1), dt=0.1, desc="Interpolating Maps: ", showspeed=true, enabled=verbose)
  @floop for h in 1:size(maps, 1)
    w[:, h] .= regionmean(maps[h, :, :, :], xyz, K, R)
    # next!(progBar)
  end
  # finish!(progBar)
  return w
end

"""
    ball_gaussian3D(σ=2, r=2)

Construct a Gaussian kernel restricted to a ball of radius `r`.
"""
function ball_gaussian3D(σ::Real=2, r::Int=2)
  # σ=gaussian std  ;  r=ball radius

  # creating ball
  bm = ball_mask(r)[2]
  sbm = size(bm, 1)

  # computing kernel size
  R = Int(ceil(4 * σ))
  L = 2 * R + 2
  L += sbm

  # getting ball indices in kernel space
  ball_idx = [[x[1], x[2], x[3]] for x in findall(bm .== 1)]
  ball_idx = permutedims(hcat(ball_idx...))
  ball_idx .+= ((L - sbm) / 2)
  ball_idx = ball_idx #.- 0.5

  # 3D gaussian
  gauss(x, y, z, x0, y0, z0) = exp(-((x - x0)^2 + (y - y0)^2 + (z - z0)^2) / (2 * σ^2))

  # merging gaussians
  xs, ys, zs = [1:L for i in 1:3]
  Gs = [
    [gauss(x, y, z, ball_idx[i, 1], ball_idx[i, 2], ball_idx[i, 3]) for x in xs, y in ys, z in zs]
    for i in 1:size(ball_idx, 1)]
  G = sum(Gs)
  return Float32.(G ./ sum(bm))
end

"""
    interpolation(maps::AbstractArray{T,4}, xyz, σ, R, box; verbose=true)
    interpolation(map::AbstractArray{T,3}, xyz, σ, R, box; verbose=true)

Interpolate map values at positions `xyz` using a ball-Gaussian kernel.
"""
function interpolation(maps::AbstractArray{T,4}, xyz::AbstractMatrix, σ::Real, R::Int, box::BoxAround; verbose=true) where {T<:AbstractFloat}
  K = ball_gaussian3D(σ, R)
  L = size(K, 1)
  l = Int(floor(L / 2))
  cint = int_coords(xyz, box)
  return regionmean(maps, cint, K, l; verbose)
end
function interpolation(maps::AbstractArray{T,3}, xyz::AbstractMatrix, σ::Real, R::Int, box::BoxAround; verbose=true) where {T<:AbstractFloat}
  K = ball_gaussian3D(σ, R)
  L = size(K, 1)
  l = Int(floor(L / 2))
  cint = int_coords(xyz, box)
  return regionmean(maps, cint, K, l)
end
# ball gaussian interpolation ________________________________________________


"""
    Maps

Container for spatial maps and associated interpolation parameters.
Fields:
- `scaling` – spatial scaling factor
- `box`     – bounding [`BoxAround`](@ref)
- `radius`  – kernel radius in original units
- `σ`       – Gaussian standard deviation in original units
- `maps`    – stored maps
- `β`       – optimal bias factors
"""
struct Maps
  scaling::Float64
  box::BoxAround
  radius::Int
  σ::Float64
  maps::Array{Float32}
  β::Vector{Float64}

  function Maps(
    coords::AbstractMatrix,
    X::AbstractArray;
    scaling::Float64=2.0, padding::Float64=0.1,
    R::Int=4,
    σ::Float64=4.0,
    verbose=true
  )
    # scaling space
    R = Int(round(R / scaling))
    σ = σ / scaling
    coords = coords ./ scaling

    # building map
    box = BoxAround(coords; padding)
    nmaps = create_map(coords, X, box; R, verbose)

    A = interpolation(nmaps, coords, σ, R, box; verbose)
    bA = find_optimal_bias(X, A, minbias=0, maxbias=1, stepbias=0.005)

    new(scaling, box, Int(R * scaling), σ * scaling, nmaps, bA)
  end

  function Maps(
    coords::AbstractMatrix,
    box::BoxAround,
    X::AbstractArray;
    scaling::Float64=2.0,
    R::Int=4,
    σ::Float64=4.0,
    verbose=true
  )
    # scaling space
    R = Int(round(R / scaling))
    σ = σ / scaling
    coords = coords ./ scaling

    # building map
    nmaps = create_map(coords, X, box; R, verbose)

    A = interpolation(nmaps, coords, σ, R, box; verbose)
    bA = find_optimal_bias(X, A, minbias=0, maxbias=1, stepbias=0.005)

    new(scaling, box, Int(R * scaling), σ * scaling, nmaps, bA)
  end

  function Maps(grp::HDF5.Group)
    new(
      grp["scaling"][],
      BoxAround(
        grp["box/lims"][],
        grp["box/size"][],
        grp["box/origin"][],
      ),
      grp["radius"][],
      grp["sigma"][],
      grp["maps"][],
      grp["beta"][],
    )
  end

  function Maps(
    other::Maps,
    coords::AbstractMatrix,
    X::AbstractArray;
    verbose=true
  )
    # getting params from other
    scaling = other.scaling
    box = other.box
    radius = other.radius
    σ = other.σ

    # scaling space
    R = Int(round(radius / scaling))
    σ = σ / scaling
    coords = coords ./ scaling

    # building map
    nmaps = create_map(coords, X, box; R, verbose)

    A = interpolation(nmaps, coords, σ, R, box; verbose)
    bA = find_optimal_bias(X, A, minbias=0, maxbias=1, stepbias=0.005)

    new(scaling, box, Int(R * scaling), σ * scaling, nmaps, bA)
  end


end

"""
    dump_maps(grp::HDF5.Group, M::Maps; comment="")
    dump_maps(fid::HDF5.File, M::Maps, name; comment="")
    dump_maps(filename::String, M::Maps, name; comment="") -> String

Persist [`Maps`] to HDF5 storage.
"""
function dump_maps(grp::HDF5.Group, M::Maps; comment::String="")
  attrs(grp)["comment"] = comment
  grp["scaling"] = M.scaling
  grp["radius"] = M.radius
  grp["sigma"] = M.σ
  grp["maps"] = M.maps
  grp["beta"] = M.β
  grp_box = create_group(grp, "box")
  grp_box["lims"] = M.box.lims
  grp_box["size"] = M.box.size
  grp_box["origin"] = M.box.origin
end
function dump_maps(fid::HDF5.File, M::Maps, name::String; comment::String="")
  grp = create_group(fid, name)
  dump_maps(grp, M; comment)
end
function dump_maps(filename::String, M::Maps, name::String; comment::String="")
  fid = h5open(filename, "cw")
  dump_maps(fid, M, name; comment)
  close(fid)
  return filename
end


"""
    interpolation(M::Maps, coords; verbose=true)

Interpolate stored maps at `coords` and apply bias factors.
"""
function interpolation(M::Maps, coords::AbstractMatrix; verbose=true)
  A = interpolation(
    M.maps,
    coords ./ M.scaling,
    M.σ / M.scaling,
    Int(M.radius / M.scaling),
    M.box;
    verbose
  )
  return A .* M.β'
end


"""
    find_optimal_bias(trueX, newX; minbias=0, maxbias=10, stepbias=0.1)

Return the multiplicative bias minimizing nRMSE between `trueX` and `newX`.
Works on vectors and matrices.
"""
function find_optimal_bias(trueX::AbstractVector, newX::AbstractVector; minbias::Real=0, maxbias::Real=10, stepbias::Real=0.1)
  opt_bias(b) = nRMSE(trueX, newX * b)
  biases = minbias:stepbias:maxbias
  rl = [opt_bias(b) for b in biases]
  return biases[argmin(rl)]
end
function find_optimal_bias(trueX::AbstractMatrix, newX::AbstractMatrix; minbias::Real=0, maxbias::Real=10, stepbias::Real=0.1)
  biases = Vector{Float64}(undef, size(trueX, 2))
  @floop for i in 1:size(trueX, 2)
    biases[i] = find_optimal_bias(trueX[:, i], newX[:, i]; minbias, maxbias, stepbias)
  end
  return biases
end




"""
    KL_divergence(p, q)

Kullback–Leibler divergence between distributions `p` and `q`.
"""
function KL_divergence(p, q)
  return sum(p .* log2.(p ./ q))
end

"""
    JS_divergence(p, q)

Jensen–Shannon divergence between `p` and `q`.
"""
function JS_divergence(p, q)
  m = (p .+ q) ./ 2
  kl_pm = KL_divergence(p, m)
  kl_qm = KL_divergence(q, m)
  return 0.5 * kl_pm + 0.5 * kl_qm
end

"""
    JS_distance(p, q)
    JS_distance(ps::Array{Float32,4})

Jensen–Shannon distance for two distributions or pairwise distances for a
stack of distributions.
"""
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
