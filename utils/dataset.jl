"""
    Data

Structure to save data used to train an RBM.

### Fields

- `name`        -- name of the dataset
- `spikes`      -- matrix of spikes (or voxel activities)
- `coords`      -- coordinates of neurons (or voxels)
"""
struct Data
    name::String
    spikes::Union{BitMatrix,Matrix}
    coords::Matrix
end
