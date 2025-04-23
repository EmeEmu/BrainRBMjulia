@recipe(IdPlotter) do scene
  Attributes(
    id_color=:gray,
    color=(:black, 0.75),
    scale=log10,
    cmap=:plasma,
    bins=100,
    switch_thresh=1.e3,
    nrmse=nothing,
  )
end


function Makie.plot!(ip::IdPlotter{<:Tuple{AbstractArray,AbstractArray}})
  x, y = lift(vec, ip[1]), lift(vec, ip[2])
  lx, ly = length(x.val), length(y.val)
  mmin = @lift(min(minimum($x), minimum($y)))
  mmax = @lift(max(maximum($x), maximum($y)))
  range = @lift([$mmin, $mmax])

  lines!(ip, range, range, color=ip.color)

  if lx < ip.switch_thresh.val
    scatter!(ip, x, y, color=ip.color)
  else
    h = hexbin!(ip, x, y, bins=ip.bins, colormap=ip.cmap, colorscale=ip.scale)
    ip.colorrange = h.colorrange
    ip.colormap = h.colormap
  end

  if isa(ip.nrmse.val, Nothing)
    nrmse = @lift(nRMSE($x, $y))
  else
    nrmse = ip.nrmse
  end
  text!(
    ip,
    mmax,
    mmin,
    text=@lift("nRMSE = $(round($nrmse, sigdigits=3))"),
    align=(:right, :bottom),
    space=:data,
  )
  ip
end


function multi_id(data::AbstractArray; panelsize::Int=100, xlabel="X", ylabel="Y", layout::Symbol=:br, loglog=false, switch_thresh=1.e3)
  n = size(data, 1)
  mmax = maximum(data)
  mmin = minimum(data)
  if loglog
    mmin = 10^(log10(mmin) - 1)
  end

  if layout == :br
    figsize = (panelsize, panelsize) .* (n - 1)
  elseif layout == :col
    ni = Integer(n * (n - 1) / 2)
    figsize = (panelsize, panelsize) .* (1.2, ni)
  elseif layout == :row
    ni = Integer(n * (n - 1) / 2)
    figsize = (panelsize, panelsize) .* (ni, 1.2)
  end
  fig = Figure(size=figsize)

  plots = []
  for i in 1:n
    for j in 1:n
      if i < j
        if layout == :br
          pos = fig[n-i, j-1]
        elseif layout == :col
          pos = fig[condenced_inds(n, i, j), 1]
        elseif layout == :row
          pos = fig[1, condenced_inds(n, i, j)]
        end
        ax = Axis(
          pos, title="$(i),$(j)",
          limits=(mmin, mmax, mmin, mmax), aspect=1,
          xticksvisible=false, xticklabelsvisible=false,
          yticksvisible=false, yticklabelsvisible=false,
        )
        if loglog
          ax.xscale = log10
          ax.yscale = log10
        end
        h = idplotter!(ax, data[i, :, :], data[j, :, :]; switch_thresh)
        push!(plots, h)
        if (i == 1) & (j == 2)
          ax.xticksvisible = true
          ax.xticklabelsvisible = true
          ax.yticksvisible = true
          ax.yticklabelsvisible = true
          ax.xlabel = xlabel
          ax.ylabel = ylabel
        end
      end
    end
  end

  # if colorbar needed
  if any([hasproperty(h, :colormap) for h in plots])
    # normalising all colormaps
    ranges = [h.colorrange.val[k] for k in 1:2, h in plots]
    cmin, cmax = minimum(ranges[1, :]), maximum(ranges[2, :])
    for h in plots
      h.colorrange = (cmin, cmax)
    end

    # making label
    scalee = plots[1].scale.val
    if scalee == log10
      label = "Log Density"
    elseif scalee == identity
      label = "Density"
    else
      label = "???????"
    end

    # plotting colorbar
    cb = Colorbar(fig[2, 3], plots[1], label=label, halign=:right)
    resize!(fig.scene, figsize .+ (cb.axis.protrusion.val, 0))

  end

  # adjusting layout
  rowgap!(fig.layout, Fixed(0))
  colgap!(fig.layout, Fixed(0))
  resize_to_layout!(fig)
  return fig
end
function multi_id(data::Vector{Matrix{Float32}}; panelsize::Int=100, xlabel="X", ylabel="Y", layout::Symbol=:br, loglog=false, switch_thresh=1.e3)
  return multi_id(
    vecmatTOarray(data);
    panelsize, xlabel, ylabel, layout, loglog, switch_thresh
  )
end
function multi_id(data::Vector{Matrix{Float64}}; panelsize::Int=100, xlabel="X", ylabel="Y", layout::Symbol=:br, loglog=false, switch_thresh=1.e3)
  return multi_id(
    vecmatTOarray(data);
    panelsize, xlabel, ylabel, layout, loglog, switch_thresh
  )
end
function multi_id(data::Vector{Vector{Float64}}; panelsize::Int=100, xlabel="X", ylabel="Y", layout::Symbol=:br, loglog=false, switch_thresh=1.e3)
  return multi_id(
    vecvecTOarray(data);
    panelsize, xlabel, ylabel, layout, loglog, switch_thresh
  )
end
