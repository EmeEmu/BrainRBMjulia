# BrainRBMjulia

BrainRBMjulia builds on the [RestrictedBoltzmannMachines.jl](https://github.com/cossio/RestrictedBoltzmannMachines.jl) implementation and adapts it for whole-brain zebrafish calcium imaging data. The package wraps core RBM routines, adds domain-specific utilities, and provides custom visualizations for analyzing neural activity.

## Repository structure

The codebase is organized into two main folders:

* `src/` – Core RBM functionality and utilities.
  * `rbms.jl` defines model types and constructors.
  * `train.jl`, `generate.jl`, and `state_sampling.jl` handle model training and data generation.
  * `maps.jl` and `statistics.jl` implement evaluation metrics and map-building routines.
  * `saving.jl` and `extendables.jl` offer serialization helpers and callbacks.
  * `utils/` contains helper modules for binary encodings, coupling approximations, dataset handling, error metrics, reordering, and voxel manipulation.
* `ext/` – [Makie](https://makie.juliaplots.org/) extensions and plotting recipes.
  * `BrainMakieExt.jl` registers Makie methods used by the package.
  * Additional files provide recipes for brain projections, correlation heatmaps, RBM diagrams, and other visual diagnostics.

An `example/` directory illustrates how to train models and visualize results.

## Usage

Add the package in a Julia environment and load it with:

```julia
using Pkg
Pkg.add(path="/path/to/BrainRBMjulia")
using BrainRBMjulia
```

Most workflows revolve around building `BrainRBM` models, training on calcium imaging datasets, and using the Makie extensions for visual exploration.

## Testing

Run the package tests with:

```julia
julia --project -e 'using Pkg; Pkg.test()'
```

The test suite exercises sampling, map construction, serialization, and plotting utilities.

