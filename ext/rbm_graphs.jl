function generate_energy_plotter(rbm, gen::GeneratedData, data::DatasetSplit)
  fig = Figure(size=(2 * 200, 200))

  ax = Axis(fig[1, 1], title="Thermalisation", xlabel="Thermalisation Steps", ylabel="mean Free Energy")
  lines!(ax, gen.thermal, color=:black)

  ax = Axis(fig[1, 2], title="Energies", xlabel="Datasets", ylabel="Free Energy")
  VALS = [data.train, data.valid, gen.v]#, rand(Bool,size(spikes))]
  names = ["train", "valid", "generated"]#, "random"]
  vals = reduce(vcat, [free_energy(rbm, v) for v in VALS])
  labs = reduce(vcat, [fill(names[i], size(VALS[i])[2]) for i in 1:1:length(VALS)])
  rainclouds!(ax, labs, vals, plot_boxplots=false, gap=0.1, cloud_width=1.5)

  colsize!(fig.layout, 1, Relative(0.75))
  return fig
end

function stats_plotter(moments::MomentsAggregate; nrmses::Union{Dict,Nothing}=nothing)
  stats = ["<v>", "<h>", "<vh>", "<vv> - <v><v>", "<hh> - <h><h>"]

  if isnothing(nrmses)
    println("recomputing nRMSEs")
    nrmses = nRMSE_from_moments(moments)
  end

  fig = Figure(size=(3 * 200, 2 * 200))
  i = 1
  j = 1
  for a in stats
    ax = Axis(fig[j, i], title=a, xlabel="data", ylabel="generated")
    h = idplotter!(
      ax,
      moments.valid[a],
      moments.gen[a],
      nrmse=nrmses[a],
    )
    if i <= 2
      i = i + 1
    else
      i = 1
      j = j + 1
    end
  end

  return fig
end

function hu_params_plotter(pos::GridPosition, layer::xReLU)
  g = GridLayout(pos)

  x = zeros(size(layer, 1))

  ax = Axis(g[1, 1], xticklabelsvisible=false, xticksvisible=false, title="Δ")
  rainclouds!(ax, x, layer.Δ, plot_boxplots=false)

  ax = Axis(g[1, 2], xticklabelsvisible=false, xticksvisible=false, title="γ")
  rainclouds!(ax, x, layer.γ, plot_boxplots=false)

  ax = Axis(g[1, 3], xticklabelsvisible=false, xticksvisible=false, title="θ")
  rainclouds!(ax, x, layer.θ, plot_boxplots=false)

  ax = Axis(g[1, 4], xticklabelsvisible=false, xticksvisible=false, title="ξ")
  rainclouds!(ax, x, layer.ξ, plot_boxplots=false)
  colgap!(g, Relative(0.05))

  Label(g[1, 1:4, Top()], "Potential Parameters", valign=:bottom,
    font=:bold,
    padding=(0, 0, 15, 0),
    fontsize=8,
  )
  Label(g[1, 1:4, Bottom()], "Densities", valign=:top,
    padding=(0, 0, 0, 5),
    fontsize=7,
  )

  # return fig
end

function hidden_hists(pos::GridPosition, h_data, h_gen)
  M = size(h_data, 1)
  T1 = size(h_data, 2)
  T2 = size(h_gen, 2)
  n = M * T1 + M * T2
  d = T1 + T2

  xs, ys, sides = zeros(Int32, n), zeros(Float32, n), zeros(Int16, n)
  for h in 0:1:M-1
    i = h * d + 1
    j = h * d + T1
    k = h * d + T1 + 1
    l = (h + 1) * d

    xs[i:j] .= ones(Int32, T1) .* (h + 1)
    xs[k:l] .= ones(Int32, T2) .* (h + 1)

    ys[i:j] .= h_data[h+1, :]
    ys[k:l] .= h_gen[h+1, :]

    sides[i:j] .= ones(Int32, T1) .* 1
    sides[k:l] .= ones(Int32, T2) .* 2
  end

  colors = @. ifelse(sides == 1, :grey, :orange)
  sides = @. ifelse(sides == 1, :left, :right)

  l = maximum(abs.(quantile(ys, [0.01, 0.99])))

  ax = Axis(pos, title="Hidden Values per HU", xlabel="Hidden Units", ylabel="Hidden Values")
  violin!(ax, xs, ys, side=sides, color=colors, gap=0.01, datalimits=(-l, l), width=2)
  # return fig
end



# function misc_plots(rbm::Union{RBM,StandardizedRBM}, spikes::AbstractArray)
#   I_vh = inputs_h_from_v(rbm, spikes)
#   h_trans = translate(rbm, spikes)
#   C = cor(h_trans')
#
#   fig = Figure(size=CURRENT_THEME.size.val .* (4, 3))
#
#
#   ax_w = Axis(fig[1, 1], title="Weights", yscale=log10, xlabel=L"w_{i,j}", ylabel="Density")
#   hist!(ax_w, vec(rbm.w), offset=1.e-0, bins=100, color=:black, normalization=:density)
#
#   ax_I = Axis(fig[1, 2], title="Inputs to Hidden", xlabel=L"I_{μ}", ylabel="Density")
#   hist!(ax_I, vec(I_vh), offset=1.e-0, bins=100, color=:black, normalization=:density)
#
#   ax_Ph = Axis(fig[1, 3], title="Hidden Values", ylabel="P(h)", xlabel="h")
#   density!(ax_Ph, vec(sample_from_inputs(rbm.hidden, I_vh)), label="sampled from inputs", color=:grey)
#   density!(ax_Ph, vec(h_trans), label="translated", color=(:orange, 0.5))
#   #axislegend(ax_Ph, position=:rt, framevisible)
#
#   ax_mh = Axis(fig[2, 2], title="Hidden Mean", xlabel="Mean h", ylabel="#")
#   hist!(ax_mh, vec(mean(h_trans, dims=2)), bins=10, color=:black)
#   ax_vh = Axis(fig[2, 3], title="Hidden Variance", xlabel="Var h", ylabel="#")
#   hist!(ax_vh, vec(var(h_trans, dims=2)), bins=10, color=:orange)
#
#   ax_C = Axis(fig[3, 3], title="Hidden Correlations", xlabel="Hidden Units", ylabel="Hidden Units", aspect=1)
#   h = corrplotter!(ax_C, C)
#   Colorbar(fig[3, 4], h, label="Pearson Correlation")
#
#   hu_params_plotter(fig[2, 1], rbm.hidden)
#
#   #hidden_hists(fig[3,1:2], h_trans, gen.h)
#
#   ax_hact = Axis(fig[4, 1:3], title="Hidden Activity", xlabel="Time (frames)", ylabel="Hidden Units")
#   l = quantile_range(h_trans)
#   h = heatmap!(h_trans', colormap=:berlin, colorrange=(-l, +l))
#   Colorbar(fig[4, 4], h, label="Hidden Value")
#   return fig
# end;


