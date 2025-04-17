### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 1b16956a-193c-11f0-1515-af304ea28870
begin
	import Pkg
	Pkg.activate(Base.current_project())
	using BrainRBMjulia
	using PlutoUI
	using GLMakie
	using HDF5
	using Clustering: hclust
end

# ╔═╡ 1bf0e00f-e30b-4dab-b4a6-9a871626c588
TableOfContents()

# ╔═╡ cd361452-e578-444a-bcee-df3501be60c0


# ╔═╡ 8a6abb9d-72be-4d1d-9833-752cf52f3d11
md"""
# 1. Loading Example Data

We will load one fish from the example dataset provided with this package (it contains 3 fish in total).

The example data is a 2D slice of a zebrafish brain recorded during ~6mins at 2.5Hz.
"""

# ╔═╡ f2eb85f4-e65c-40be-bcfd-528d86781196
exampleset_path = "2d_example_dataset.h5";

# ╔═╡ d66f74fb-fa4a-411a-9457-b70e8e238339
spikes = Bool.(h5read(exampleset_path, "Fish1/spikes"));

# ╔═╡ df54ea7e-3984-4b46-98b3-cdb71ae42de0
coords = permutedims(h5read(exampleset_path, "Fish1/coords"));

# ╔═╡ e9ff6c9d-64f0-4c2b-8b81-00e7e3e1f2fb
md"N neurons = $(size(spikes,1)) | T frames = $(size(spikes,2))"

# ╔═╡ 42b9ffa4-779a-4ba9-a0f6-4db7a481334e
md"To simplify visualization, we will reorder the neurons based on their pairwise correlations"

# ╔═╡ 069fcbb8-95af-4087-a98e-b3fdb9e0f39e
begin
	C_visibles = cor(spikes');
	C_visibles[isnan.(C_visibles)] .= 0.
	HCLUST_visible = hclust(1 .- C_visibles, linkage=:ward, branchorder=:optimal)
	order_visible = HCLUST_visible.order
end;

# ╔═╡ 301821fc-aa3c-4e5c-80ba-5e918e9aac1e
begin
	spikes .= spikes[order_visible,:]
	coords .= coords[order_visible,:]
end;

# ╔═╡ 5b0c04ed-1e8c-4ccd-b2b4-eea866b7ba1e


# ╔═╡ 1478eb70-5929-4585-8364-d6c17814cd3c
begin
	fig_dataset = Figure(size=(1000,500))
	
	Axis(
		fig_dataset[1,1], 
		aspect=DataAspect(), 
		title = "Spatial coordinates of neurons",
		leftspinevisible=false, rightspinevisible=false, topspinevisible=false, bottomspinevisible=false,
		xgridvisible=false, ygridvisible=false,
		xticksvisible=false, yticksvisible=false,
		xticklabelsvisible=false, yticklabelsvisible=false,
	)
	scatter!(coords[:,2], -coords[:,1], color=(:black, 0.2), markersize=5)
	lines!([0,200],[-700,-700], linewidth=5, color=:black)
	text!(100,-695, text="200μm", align = (:center, :bottom))

	Axis(
		fig_dataset[1,2:4],
		title="Binarized Spikes trains",
		xlabel="Time (frame)",
		ylabel="Neurons",
	)
	heatmap!(spikes')
	fig_dataset
end

# ╔═╡ 3d31c0b1-3b0e-4a70-bd19-fa9e21ffaea1


# ╔═╡ 3157a23f-3421-4602-982d-3dd69e6ca2b0


# ╔═╡ 0ba34e65-eddf-4c2d-8542-cb9d1338d6aa
md"""
## 1.2. Building dataset
"""

# ╔═╡ 7c579f03-1eec-4309-9988-bf61f0e41dbd
# building dataset stucture
dataset = Data(
	"Example 2D Fish",
	spikes,
	coords,
);

# ╔═╡ 7fa1a84e-a5d9-42ae-8250-fa4c57ea0215
md"""
## 1.3. Save dataset to disk
"""

# ╔═╡ bad66f81-bc6b-4e16-8200-b184bee31c17
saved_dataset_path = tempname() .* ".h5"

# ╔═╡ 16475b74-d8fe-488c-97d3-70c4081da5d1
dump_data(saved_dataset_path, dataset, comment="fake dataset");

# ╔═╡ e12fb41f-e074-4367-893d-c7fdf8426d7a


# ╔═╡ d977060e-6d26-408e-96a2-e81328a362f1
md"""
## 1.4. Split into test train
"""

# ╔═╡ 34a6b922-cecc-43df-b73f-6d9f9144a7d8
begin
	ssplt = SectionSplit(dataset.spikes, 0.7, N_vv=100);
	dsplit = split_set(dataset.spikes, ssplt, q=0.1);
end;

# ╔═╡ 4c754805-ae4e-4450-aa70-3fb784e8c268
mvtrain, mvtest, mvvtrain, mvvtest = section_moments(
	dsplit.train, 
	dsplit.valid,
);

# ╔═╡ fd0763a9-9850-4442-902b-1850e7e73f3a
nRMSE(mvtrain, mvtest), nRMSE(mvvtrain, mvvtest)

# ╔═╡ f5e6d72f-e71c-4ba7-af4f-db94dfaa5549
begin
	fig_datasplit = Figure(size=(800,400))
	
	Axis(fig_datasplit[1,1], aspect=1, xlabel="Train set", ylabel="Test set", title="⟨v⟩")
	hexbin!(mvtrain, mvtest, bins=100, colorscale=log10, colormap=:inferno)
	lines!([0,0.6], [0,0.6], color=:black)
	
	Axis(fig_datasplit[1,2], aspect=1, xlabel="Train set", ylabel="Test set", title="⟨vv⟩ - ⟨v⟩⟨v⟩")
	hexbin!(vec(mvvtrain), vec(mvvtest), bins=100, colorscale=log10, colormap=:inferno)
	lines!([-0.2,0.3], [-0.2,0.3], color=:black)
	
	fig_datasplit
end

# ╔═╡ 0c5d5cdb-ccec-424f-8685-97c681b3b518


# ╔═╡ f541b841-bb26-49b4-b5e8-08f7291a9e2f


# ╔═╡ ae16d3c6-33f6-445f-b4b1-a0a3b1bb86dc
md"""
# 2. RBM Training
"""

# ╔═╡ 92ba66d1-01c4-42b0-b678-3c95a1febfdb
md"""
## 2.1. Prepare RBM

We'll start by creating a Standardized RBM with Binary visible units and `M` xReLU hidden units. A preset already exists and can by optained with the `BrainRBM` function.
"""

# ╔═╡ dcc86d75-0c18-493a-b6f3-14db0a83d38a
M = 5; # Number of hidden units

# ╔═╡ 487c4c58-e3be-4b41-a024-f4a9b0a63d90
rbm = BrainRBM(dsplit.train, M);

# ╔═╡ 69fc254d-611c-4ac9-992b-2da1b7505ab1
md"""
## 2.2. Train
"""

# ╔═╡ bd6cef85-7d12-45bf-94ad-4da62ed79119
history, params = training_wrapper(rbm, dsplit.train, iters=1000, record_ps=true, verbose=true, batchsize=25);

# ╔═╡ ba2bf38d-3540-42f6-a8e7-d7efa4f408be
begin
	fig_training = Figure()
	
	Axis(fig_training[1,1], xlabel="Training iterations", ylabel="Learning Rate")
	lines!(history[:lr].iterations, history[:lr].values)
	xlims!(0, params["iters"])
	
	Axis(fig_training[2,1], xlabel="Training iterations", ylabel="Lop Pseudo-Likelyhood")
	lines!(history[:lpl].iterations, history[:lpl].values)
	xlims!(0, params["iters"])
	
	fig_training
end

# ╔═╡ 87e5c9a1-24a1-4816-9f57-a8acac963493


# ╔═╡ a88be571-bf95-4a0d-981c-76ab438f6697


# ╔═╡ e140ec64-26ca-4945-92b6-810f490b6850
md"""
## 2.3. Aesthetical changes
"""

# ╔═╡ 8ea89877-1199-410b-b5e1-a114e85d9737
swap_hidden_sign!(rbm);

# ╔═╡ 750589a9-17ff-4339-b838-15d4cbfdfc5b
reorder_hus!(rbm, dsplit.train);

# ╔═╡ f19fa0cc-e715-4d31-bb6c-95c656db1302
md"""
## 2.4. Generating data
"""

# ╔═╡ 971bc374-1a3e-4641-8822-3337328d9161
gen = gen_data(rbm, nsamples=500, nthermal=50, nstep=10);

# ╔═╡ 6baa1301-e3ff-49be-8983-4d926fe134ce
begin
	fig_gen = Figure(size=(2 * 400, 400))
	
	Axis(fig_gen[1, 1], title="Thermalisation", xlabel="Thermalisation Steps", ylabel="mean Free Energy")
	lines!(gen.thermal, color=:black)
	
	Axis(fig_gen[1, 2], title="Energies", xlabel="Datasets", ylabel="Free Energy")
	VALS = [dsplit.train, dsplit.valid, gen.v]#, rand(Bool,size(spikes))]
	names = ["train", "valid", "generated"]#, "random"]
	vals = reduce(vcat, [free_energy(rbm, v) for v in VALS])
	labs = reduce(vcat, [fill(names[i], size(VALS[i])[2]) for i in 1:1:length(VALS)])
	rainclouds!(labs, vals, plot_boxplots=false, gap=0.1, cloud_width=1.5)
	
	colsize!(fig_gen.layout, 1, Relative(0.75))
	fig_gen
end

# ╔═╡ 9305f465-aee5-4481-a4f7-f00b612f0059
md"""
## 2.5. Evaluating RBM
"""

# ╔═╡ 7d97bacc-7dff-43b5-b76a-b8abef7aa2a4
moments = compute_all_moments(rbm, dsplit, gen);

# ╔═╡ af380472-e163-47af-90a7-f3a4926bc6b5
nrmses = nRMSE_from_moments(moments)

# ╔═╡ 4f595730-5356-43a7-9966-f5a72f033750
begin
	stats = ["<v>", "<h>", "<vh>", "<vv> - <v><v>", "<hh> - <h><h>"]
	fig_stats = Figure(size=(5*200, 1*200))
	  for (i,a) in enumerate(stats)
	    ax = Axis(fig_stats[1, i], title=a, xlabel="data", ylabel="generated", aspect=1)
		  x = vec(moments.valid[a])
		  y = vec(moments.gen[a])
	    h = hexbin!(
	      x,
	      y,
		  bins=100,
		  colormap=:inferno,
		  colorscale=log10
	    )
		  amax = max(maximum(x), maximum(y))
		  amin = min(minimum(x), minimum(y))
		  lines!([amin, amax], [amin, amax], color=:black)
	  end

	fig_stats
end

# ╔═╡ ed4b1ab8-8dbf-4f78-85cf-10cd56682550


# ╔═╡ 34fd81fa-bf8e-46b8-b58f-91a4e51de5da
md"""
## 2.6. Save rbm to disk
"""

# ╔═╡ 5e1a4541-a4fd-4acd-8f39-b7805e2e4907
saved_rbm_path = tempname() .* ".h5"

# ╔═╡ 39bc7d58-fdb9-439d-bf7a-ba3a5ef6daf0
dump_brainRBM(
				saved_rbm_path, 
				rbm, params, 
				nrmses, 
				dsplit, gen, 
				translate(rbm, dataset.spikes) ; 
				comment="Example rbm",
			)

# ╔═╡ db5ccd37-1e8a-4683-88dc-3c5190ed6a8f


# ╔═╡ 8af5299e-44d9-4fe9-b10e-3b529270184d


# ╔═╡ becf4db4-e6e6-4208-87f0-3cc23544202b
md"""
# 3. Investigate RBM
"""

# ╔═╡ 5be66eae-99a8-4f08-8abf-5fccb4eb4c55
md"""
## 3.1. Weights
"""

# ╔═╡ 723d6615-5a1d-44e8-8c92-6b8667deef41
begin
	fig_weights = Figure(size=(2*400,400))
	l = quantile(vec(abs.(rbm.w)), 0.99)

	Axis(fig_weights[1,1], xlabel="Weights", ylabel="Density", yscale=log10)
	hist!(
		vec(rbm.w), 
		normalization=:pdf, offset=1.e-5, bins=100,
	)

	Axis(fig_weights[1,2], xlabel="Neurons", ylabel="Hidden Units")
	heatmap!(rbm.w, colormap=:seismic, colorrange=(-l,+l))
	fig_weights
end

# ╔═╡ 8a75694d-d47e-4991-9934-d3a9fc081112
md"Hidden Unit μ $(@bind μ PlutoUI.Slider(1:M))"

# ╔═╡ 9f9f2ea5-6f7b-4e79-8060-f07c30a25ab7
begin
	fig_weight2D = Figure()
	Axis(fig_weight2D[1,1], aspect=DataAspect())
	scatter!(
		dataset.coords[:,2], -dataset.coords[:,1], 
		color=rbm.w[:,μ], colormap=:seismic, colorrange=(-l,+l),
		markersize=4
	)
	fig_weight2D
end

# ╔═╡ dea797e0-2230-4d5a-87bc-350c281698ed


# ╔═╡ de6d97c2-28b8-4225-87c7-adb9176ba269
md"""
## 3.2. Hidden Activity
"""

# ╔═╡ 1cc2ce2d-5eb5-4825-8e9e-e4a0d0d8939d
heatmap(translate(rbm, dataset.spikes)', colormap=:berlin)

# ╔═╡ a6347ed3-e31b-4116-bc4f-21f0c3f354b0


# ╔═╡ 385b2673-51f0-4f6f-a119-33a85d26a2dc


# ╔═╡ Cell order:
# ╠═1b16956a-193c-11f0-1515-af304ea28870
# ╠═1bf0e00f-e30b-4dab-b4a6-9a871626c588
# ╠═cd361452-e578-444a-bcee-df3501be60c0
# ╟─8a6abb9d-72be-4d1d-9833-752cf52f3d11
# ╠═f2eb85f4-e65c-40be-bcfd-528d86781196
# ╠═d66f74fb-fa4a-411a-9457-b70e8e238339
# ╠═df54ea7e-3984-4b46-98b3-cdb71ae42de0
# ╟─e9ff6c9d-64f0-4c2b-8b81-00e7e3e1f2fb
# ╟─42b9ffa4-779a-4ba9-a0f6-4db7a481334e
# ╠═069fcbb8-95af-4087-a98e-b3fdb9e0f39e
# ╠═301821fc-aa3c-4e5c-80ba-5e918e9aac1e
# ╠═5b0c04ed-1e8c-4ccd-b2b4-eea866b7ba1e
# ╟─1478eb70-5929-4585-8364-d6c17814cd3c
# ╠═3d31c0b1-3b0e-4a70-bd19-fa9e21ffaea1
# ╠═3157a23f-3421-4602-982d-3dd69e6ca2b0
# ╟─0ba34e65-eddf-4c2d-8542-cb9d1338d6aa
# ╠═7c579f03-1eec-4309-9988-bf61f0e41dbd
# ╟─7fa1a84e-a5d9-42ae-8250-fa4c57ea0215
# ╠═bad66f81-bc6b-4e16-8200-b184bee31c17
# ╠═16475b74-d8fe-488c-97d3-70c4081da5d1
# ╠═e12fb41f-e074-4367-893d-c7fdf8426d7a
# ╟─d977060e-6d26-408e-96a2-e81328a362f1
# ╠═34a6b922-cecc-43df-b73f-6d9f9144a7d8
# ╠═4c754805-ae4e-4450-aa70-3fb784e8c268
# ╠═fd0763a9-9850-4442-902b-1850e7e73f3a
# ╟─f5e6d72f-e71c-4ba7-af4f-db94dfaa5549
# ╠═0c5d5cdb-ccec-424f-8685-97c681b3b518
# ╠═f541b841-bb26-49b4-b5e8-08f7291a9e2f
# ╟─ae16d3c6-33f6-445f-b4b1-a0a3b1bb86dc
# ╟─92ba66d1-01c4-42b0-b678-3c95a1febfdb
# ╠═dcc86d75-0c18-493a-b6f3-14db0a83d38a
# ╠═487c4c58-e3be-4b41-a024-f4a9b0a63d90
# ╟─69fc254d-611c-4ac9-992b-2da1b7505ab1
# ╠═bd6cef85-7d12-45bf-94ad-4da62ed79119
# ╟─ba2bf38d-3540-42f6-a8e7-d7efa4f408be
# ╠═87e5c9a1-24a1-4816-9f57-a8acac963493
# ╠═a88be571-bf95-4a0d-981c-76ab438f6697
# ╟─e140ec64-26ca-4945-92b6-810f490b6850
# ╠═8ea89877-1199-410b-b5e1-a114e85d9737
# ╠═750589a9-17ff-4339-b838-15d4cbfdfc5b
# ╟─f19fa0cc-e715-4d31-bb6c-95c656db1302
# ╠═971bc374-1a3e-4641-8822-3337328d9161
# ╟─6baa1301-e3ff-49be-8983-4d926fe134ce
# ╟─9305f465-aee5-4481-a4f7-f00b612f0059
# ╠═7d97bacc-7dff-43b5-b76a-b8abef7aa2a4
# ╠═af380472-e163-47af-90a7-f3a4926bc6b5
# ╟─4f595730-5356-43a7-9966-f5a72f033750
# ╠═ed4b1ab8-8dbf-4f78-85cf-10cd56682550
# ╟─34fd81fa-bf8e-46b8-b58f-91a4e51de5da
# ╠═5e1a4541-a4fd-4acd-8f39-b7805e2e4907
# ╠═39bc7d58-fdb9-439d-bf7a-ba3a5ef6daf0
# ╠═db5ccd37-1e8a-4683-88dc-3c5190ed6a8f
# ╠═8af5299e-44d9-4fe9-b10e-3b529270184d
# ╟─becf4db4-e6e6-4208-87f0-3cc23544202b
# ╟─5be66eae-99a8-4f08-8abf-5fccb4eb4c55
# ╟─723d6615-5a1d-44e8-8c92-6b8667deef41
# ╟─8a75694d-d47e-4991-9934-d3a9fc081112
# ╟─9f9f2ea5-6f7b-4e79-8060-f07c30a25ab7
# ╠═dea797e0-2230-4d5a-87bc-350c281698ed
# ╟─de6d97c2-28b8-4225-87c7-adb9176ba269
# ╠═1cc2ce2d-5eb5-4825-8e9e-e4a0d0d8939d
# ╠═a6347ed3-e31b-4116-bc4f-21f0c3f354b0
# ╠═385b2673-51f0-4f6f-a119-33a85d26a2dc
