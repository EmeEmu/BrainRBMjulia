using JuliaGrapher
using BrainRBMjulia


function vhs_2D_scatter(
  g::GridPosition,
  X::Vector{Float64}, Y::Vector{Float64},
  v::Union{Vb,Observable{<:Vb}},
  vh::Union{V,Observable{<:V}},
  vs::Union{V,Observable{<:V}};
  showlabels=true
) where {V<:Vector{<:AbstractFloat},Vb<:BitVector}
  if showlabels
    top = 2
  else
    top = 1
  end
  scatter_axis_args = (
    aspect=DataAspect(),
    bottomspinevisible=false, leftspinevisible=false,
    xticklabelsvisible=false, yticklabelsvisible=false,
    xticksvisible=false, yticksvisible=false,
  )
  scatter_args = (
    range=(0, 1),
    radius=4,
    edgecolor=(:black, 0.1)
  )
  colorbar_args = (
    flipaxis=true, vertical=false,
  )
  ax = Axis(g[top:top+2, 1]; scatter_axis_args...)
  h = neuron2dscatter!(X, Y, v, cmap=cmap_Gbin; scatter_args...
  )
  if showlabels
    Colorbar(g[1, 1], h, label=L"$\mathbf{v}_{t}$",
      width=Relative(0.2), ticks=([0, 1], ["off", "on"]);
      colorbar_args...
    )
  end

  ax = Axis(g[top:top+2, 2]; scatter_axis_args...)
  h = neuron2dscatter!(X, Y, vh, cmap=cmap_dff; scatter_args...
  )
  if showlabels
    Colorbar(g[1, 2], h,
      label=L"$\mathbf{v}_{model} = \mathbb{E} [ \mathbf{v}|\mathbb{E}[\mathbf{h}|\mathbf{v}_{t}] ]$",
      width=Relative(0.5); colorbar_args...
    )
  end

  ax = Axis(g[top:top+2, 3]; scatter_axis_args...)
  h = neuron2dscatter!(X, Y, vs, cmap=cmap_ainferno; scatter_args...
  )
  if showlabels
    Colorbar(g[1, 3], h, label=L"$\langle \mathbf{v} \rangle_s$",
      width=Relative(0.5); colorbar_args...
    )
  end

end
