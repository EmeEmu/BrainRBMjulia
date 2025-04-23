@recipe(PolarNRMSEPlotter) do scene
  Attributes(
    origin=(0, 0),
    markersize=5,
    linewidth=1,
    color=:black,
    make_axis=true,
    heights=[0.2, 0.7, 1.2],
    ax_color=:grey,
    ax_width=1,
    ax_fontsize=4,
  )
end

function Makie.plot!(pp::PolarNRMSEPlotter{<:Tuple{Dict}})
  dic = pp[1]
  vals = @lift(Float32.(values($dic)))
  vals = @lift([map_range(v, 0, 1, pp.heights.val[1], pp.heights.val[end]) for v in $vals])
  vals = @lift([clamp(v, 0, pp.heights.val[end] * 1.1) for v in $vals])
  labels = @lift(String.(keys($dic)))
  x0, y0 = pp.origin.val

  n = length(vals.val)

  θ = collect(range(0, 2 * pi, n + 1))[1:end-1] .+ pi / 2
  if pp.make_axis.val
    x = pp.heights.val[end] .* 1.1 .* cos.(θ)
    y = pp.heights.val[end] .* 1.1 .* sin.(θ)
    lx = pp.heights.val[end] .* 1.15 .* cos.(θ)
    ly = pp.heights.val[end] .* 1.15 .* sin.(θ)
    X, Y = Float32[], Float32[]
    lθ = θ .- pi / 2
    for i in 1:n
      push!(X, 0)
      push!(X, x[i])
      push!(Y, 0)
      push!(Y, y[i])
      if lθ[i] < pi / 2 || lθ[i] > 3 * pi / 2
        text!(
          pp, x0 + lx[i], y0 + ly[i],
          text=labels.val[i],
          #text=string(round(lθ[i]./pi; digits=2)),
          align=(:center, :bottom),
          rotation=lθ[i],
          fontsize=pp.ax_fontsize
        )
      else
        text!(
          pp, x0 + lx[i], y0 + ly[i],
          text=labels.val[i],
          #text=string(round(lθ[i]./pi; digits=2)),
          align=(:center, :top),
          rotation=lθ[i] - pi,
          fontsize=pp.ax_fontsize
        )
      end
    end
    linesegments!(pp, x0 .+ X, y0 .+ Y, color=pp.ax_color, linewidth=pp.ax_width)
    θ2 = collect(range(0, 2 * pi, 1000))
    for r in pp.heights.val
      lines!(pp, x0 .+ r .* cos.(θ2), y0 .+ r .* sin.(θ2), color=pp.ax_color, linewidth=pp.ax_width)
      text!(pp, x0 + 0.02, y0 + r, text=string(round(r - pp.heights.val[1]; digits=2)), align=(:left, :bottom), fontsize=pp.ax_fontsize)
    end
  end

  @lift(push!($vals, vals.val[1]))
  push!(θ, θ[1] + 2 * pi)
  if pp.markersize.val > 0
    scatter!(pp, @lift(x0 .+ $vals .* cos.(θ)), @lift(y0 .+ $vals .* sin.(θ)), markersize=pp.markersize, color=pp.color)
  end
  if pp.linewidth.val > 0
    θ2 = collect(range(minimum(θ), maximum(θ), 1000))
    interp = @lift(LinearInterpolation(θ, $vals, extrapolation_bc=Line()))
    VALS = @lift($interp.(θ2))
    lines!(pp, @lift(x0 .+ $VALS .* cos.(θ2)), @lift(y0 .+ $VALS .* sin.(θ2)), linewidth=pp.linewidth, color=pp.color)
  end

  pp
end


@recipe(multiPolarNRMSEPlotter) do scene
  Attributes(
    origin=(0, 0),
    markersize=5,
    linewidth=1,
    cmap=:RdYlGn_9,
    cmap_max=1.0,
    default_color=:black,
    heights=[0.2, 0.7, 1.2],
    ax_color=:grey,
    ax_width=1,
    ax_fontsize=4,
  )
end

function Makie.plot!(pp::multiPolarNRMSEPlotter{<:Tuple{Vector{Dict{Any,Any}}}})
  dics = pp[1]
  n = length(dics.val)
  attrss = NamedTuple(pp.attributes)

  for i in 1:n
    eval = @lift($dics[i])
    if i == 1
      polarnrmseplotter!(pp, eval; make_axis=true, color=pp.default_color, attrss...)
    else
      polarnrmseplotter!(pp, eval; make_axis=false, color=pp.default_color, attrss...)
    end
  end

  pp
end

function Makie.plot!(pp::multiPolarNRMSEPlotter{<:Tuple{Vector{Dict{Any,Any}},Vector}})
  dics = pp[1]
  norms = pp[2]
  n = length(dics.val)
  attrss = NamedTuple(pp.attributes)
  cmap = colorschemes[pp.cmap.val]

  for i in 1:n
    eval = @lift($dics[i])
    norm = @lift(1 - $norms[i] / pp.cmap_max.val)
    color = @lift(get(cmap, $norm))
    if i == 1
      polarnrmseplotter!(pp, eval; make_axis=true, color=color, attrss...)
    else
      polarnrmseplotter!(pp, eval; make_axis=false, color=color, attrss...)
    end
  end

  pp.colormap = cmap
  pp.colorrange = (0, pp.cmap_max.val)

  pp
end

Makie.convert_arguments(P::Type{<:PolarNRMSEPlotter}, path::String) = (load_brainRBM_eval(path),)

function Makie.convert_arguments(P::Type{<:multiPolarNRMSEPlotter}, paths::Vector{String})
  evals = load_brainRBM_eval(paths)
  norm = nRMSEs_L4(evals) ./ nRMSEs_L4(evals; max=true)
  return (evals, norm)
end
