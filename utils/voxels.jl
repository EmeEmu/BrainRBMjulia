struct VoxelGrid
  origins::Vector{Float64}
  ends::Vector{Float64}
  Ns::Vector{Int64}
  voxsize::Vector{Float64}

  map::Array{Float64,3}
  goods::Vector{CartesianIndex{3}}
  voxel_composition::Matrix{Vector{Int64}}
  voxel_activities::Vector{Matrix{Float64}}
  neuron_affiliation::Vector{Vector{Int64}}

  function VoxelGrid(
    coords::Vector{Matrix{Float64}},
    acts::Vector{Matrix{Float64}};
    voxsize::Vector{Float64}=[20.0, 20.0, 20.0],
    pad::Float64=0.1
  )
    size = voxsize # size of the voxels
    orgs, ends = grid_from_points(coords; size, pad) # box around point cloud
    Ns = [length(orgs[i]:size[i]:ends[i]) for i in 1:length(orgs)] # number of voxels in each direction
    indss = project_to_grid(coords, size, orgs) # point coordinates to grid coordinates
    voxs, capacity = populate_voxels(indss, Ns) # place points in voxels + count nb of point in each voxel
    goods = select_voxels(capacity) # voxel selection criteria
    map = Array{Float32}(undef, Ns...)
    map[:, :, :] .= NaN32
    map[goods] .= 1.0
    vox_composition, vox_activities, neuron_affil = build_voxels(voxs, goods, acts)
    new(orgs, ends, Ns, voxsize, map, goods, vox_composition, vox_activities, neuron_affil)
  end
end


function grid_from_points(coords::Matrix{Float64}; size::Vector{Float64}=[20.0, 20.0, 20.0], pad::Float64=0.1)
  maxs = maximum(coords, dims=1)
  mins = minimum(coords, dims=1)
  span = maxs .- mins
  padding = span .* pad
  orgs = mins .- padding
  ends = maxs .+ padding
  return orgs[1, :], ends[1, :]
end
function grid_from_points(coords::Vector{Matrix{Float64}}; size::Vector{Float64}=[20.0, 20.0, 20.0], pad::Float64=0.1)
  return grid_from_points(vcat(coords...); size, pad)
end

function project_to_grid(coords::Matrix{Float64}, size::Vector{Float64}, origin::Vector{Float64})
  return Int.(round.((coords .- origin') ./ size'))
end
function project_to_grid(coords::Vector{Matrix{Float64}}, size::Vector{Float64}, origin::Vector{Float64})
  return [project_to_grid(c, size, origin) for c in coords]
end

function populate_voxels(vox_inds::Matrix{Int64}, ns::Vector{Int})
  VOX = Array{Vector{Int}}(undef, ns...)
  for i in 1:ns[1], j in 1:ns[2], k in 1:ns[3]
    VOX[i, j, k] = Vector{Int}()
  end

  for p in 1:size(vox_inds, 1)
    push!(VOX[vox_inds[p, :]...], p)
  end

  CAPACITY = Array{Int}(undef, ns...)
  for i in 1:ns[1], j in 1:ns[2], k in 1:ns[3]
    CAPACITY[i, j, k] = length(VOX[i, j, k])
  end

  return VOX, CAPACITY
end
function populate_voxels(vox_inds::Vector{Matrix{Int64}}, ns::Vector{Int})
  n = length(vox_inds)
  VOX = Array{Vector{Int}}(undef, n, ns...)
  CAPACITY = Array{Int}(undef, n, ns...)
  for i in 1:n
    vox, capacity = populate_voxels(vox_inds[i], ns)
    VOX[i, :, :, :] = vox
    CAPACITY[i, :, :, :] = capacity
  end
  return VOX, CAPACITY
end

function select_voxels(CAPACITY::Array{Int64,4}; thresh::Int=2)
  goods = sum(CAPACITY .>= thresh, dims=1)[1, :, :, :] .== size(CAPACITY, 1)
  return findall(goods)
end

function build_voxels(
  VOX::Array{Vector{Int64},4},
  goods::Vector{CartesianIndex{3}},
  acts::Vector{Matrix{Float64}}
)
  vox_composition = VOX[:, goods]
  F = size(vox_composition, 1)
  V = size(vox_composition, 2)
  vox_activities = [Matrix{Float64}(undef, size(acts[f], 1), V) for f in 1:F]
  neuron_affil = [zeros(Int, size(acts[f], 2)) .- 1 for f in 1:F]
  for f in 1:F
    for i in 1:V
      neurs = vox_composition[f, i]
      vox_activities[f][:, i] .= mean(acts[f][:, neurs], dims=2)
      neuron_affil[f][neurs] .= i
    end
  end
  return vox_composition, vox_activities, neuron_affil
end


function vox_to_neur_activity(t::Int, indv::Int, vox::VoxelGrid)
  act = vox.voxel_activities[indv][t, :]
  aff = vox.neuron_affiliation[indv]
  res = Vector{Float64}(undef, length(aff))
  for i in 1:length(aff)
    if aff[i] == -1
      res[i] = 0.0
    else
      res[i] = act[aff[i]]
    end
  end
  return res
end
function vox_to_neur_activity(t::Union{UnitRange{Int64},Vector{Int64}}, indv::Int, vox::VoxelGrid)
  aff = vox.neuron_affiliation[indv]
  res = Matrix{Float64}(undef, length(t), length(aff))
  for (i, tt) in enumerate(t)
    res[i, :] .= vox_to_neur_activity(tt, indv, vox)
  end
  return res
end
