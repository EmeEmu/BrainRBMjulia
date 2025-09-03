"""
    DatasetSplit

Container holding a random training/validation split of a dataset.

Fields
------
- `train_inds`: column indices used for training
- `valid_inds`: column indices used for validation
- `train`: training subset of the data
- `valid`: validation subset of the data
"""
struct DatasetSplit
  train_inds::Vector{Int}
  valid_inds::Vector{Int}
  train::Union{Matrix,BitMatrix}
  valid::Union{Matrix,BitMatrix}
end


"""
    split_set(X; p_train=0.75)

Randomly split columns of `X` into a training and validation set with a
fraction `p_train` assigned to training.

Returns a [`DatasetSplit`](@ref).
"""
function split_set(X::Union{Matrix,BitMatrix}; p_train=0.75)
  i_train = Int(floor(p_train * size(X, 2)))
  inds = range(1, size(X, 2)) |> collect
  inds = shuffle(inds)
  train_inds = inds[begin:i_train]
  valid_inds = inds[i_train:end]
  train = X[:, train_inds]
  valid = X[:, valid_inds]
  return DatasetSplit(
    train_inds,
    valid_inds,
    train,
    valid
  )
end

"""
    SectionSplit

Utility to split a time series into sections and evaluate all possible
train/test combinations of these sections.
"""
mutable struct SectionSplit
  N::Int # number of sections
  ftrain::AbstractFloat # fraction of train set
  ftest::AbstractFloat # fraction of test set
  sec_inds::Vector{Vector{Int64}} # time indices of section
  train_comb::Vector{Vector{Int64}} # all possible training sets (by sections)
  test_comb::Vector{Vector{Int64}} # all possible testing sets (by sections)
  rmse_v_comb::Vector{Float64} # <v> moment between train and test sets for all combinations
  rmse_vv_comb::Vector{Float64} # <vv> moment between train and test sets for all combinations

  """
      SectionSplit(nt::Int, ftrain; N=10)

Construct a `SectionSplit` for a signal of length `nt`, dividing it into
`N` sections and setting aside a fraction `ftrain` for training.
"""
  function SectionSplit(nt::Int, ftrain::AbstractFloat; N::Int=10)
    # create SectionSplit from number of time points
    # it will need to be rmses will need to be computed externaly

    # create the sections
    secs = collect(Iterators.partition(1:nt, (nt ÷ N) + 1))
    secs_len = [length(p) for p in secs]
    @assert sum(secs_len) == nt
    sec_inds = [collect(seq) for seq in secs]

    # create all possible train/test section combinations
    train_comb = [combinations(1:N, Int(N * ftrain))...]
    test_comb = [filter!(e -> e ∉ seq, collect(1:N)) for seq in train_comb]

    new(N, ftrain, 1 - ftrain, sec_inds, train_comb, test_comb, zeros(2), zeros(2))
  end

  """
      SectionSplit(v::M, ftrain; N=10, N_vv=5000)

Create a `SectionSplit` from data matrix `v`, immediately computing the
RMSE between sections to guide future splits.
"""
  function SectionSplit(v::M, ftrain::AbstractFloat; N::Int=10, N_vv::Int=5000) where {M<:Union{BitMatrix,AbstractMatrix}}
    ssplt = SectionSplit(size(v, 2), ftrain; N)
    rmse_v, rmse_vv = sections_to_rmses(v, ssplt; N_vv)
    ssplt.rmse_v_comb = rmse_v
    ssplt.rmse_vv_comb = rmse_vv
    return ssplt
  end
end


"""
    section_moments(split1, split2; N_vv=100)

Compute mean activity and covariance moments for two data splits.
Returns `(m_v_1, m_v_2, m_vv_1, m_vv_2)`.
"""
function section_moments(split1::M, split2::M; N_vv::Int=100) where {M<:Union{BitMatrix,AbstractMatrix}}
  m_v_1 = mean(split1, dims=2)[:, 1]
  m_v_2 = mean(split2, dims=2)[:, 1]
  ni = shuffle(1:size(split1, 1))[1:N_vv]
  m_vv_1 = cov(split1[ni, :], dims=2, corrected=false)
  m_vv_2 = cov(split2[ni, :], dims=2, corrected=false)
  return m_v_1, m_v_2, m_vv_1, m_vv_2
end

"""
    section_to_sets(i, v, ssplt)

Return the `(train, test)` matrices for the `i`-th split defined in
`ssplt`.
"""
function section_to_sets(i::Int, v::M, ssplt::SectionSplit) where {M<:Union{BitMatrix,AbstractMatrix}}
  train_inds = vcat(ssplt.sec_inds[ssplt.train_comb[i]]...)
  test_inds = vcat(ssplt.sec_inds[ssplt.test_comb[i]]...)
  train = v[:, train_inds]
  test = v[:, test_inds]
  return train, test
end

"""
    section_to_moments(i, v, ssplt; N_vv=100)

Compute moments for the `i`-th train/test split produced by `ssplt`.
"""
function section_to_moments(i::Int, v::M, ssplt::SectionSplit; N_vv::Int=100) where {M<:Union{BitMatrix,AbstractMatrix}}
  train, test = section_to_sets(i, v, ssplt)
  return section_moments(train, test; N_vv)
end

"""
    sections_to_rmses(v, ssplt; N_vv=100)

Evaluate RMSE of first- and second-order moments for all splits encoded
in `ssplt` using data `v`.
Returns vectors `(rmse_v, rmse_vv)`.
"""
function sections_to_rmses(v::M, ssplt::SectionSplit; N_vv::Int=100) where {M<:Union{BitMatrix,AbstractMatrix}}
  rmse_v = []
  rmse_vv = []
  @progress name = "Evaluating split combinations" for i in 1:length(ssplt.train_comb)
    mv_train, mv_test, mvv_train, mvv_test = section_to_moments(i, v, ssplt; N_vv)
    push!(rmse_v, rmse(mv_train, mv_test))
    push!(rmse_vv, rmse(vec(mvv_train), vec(mvv_test)))
  end
  return rmse_v, rmse_vv
end


"""
    split_set(v, ssplt; q=0.1)

Choose a train/test split from `ssplt` whose combined RMSE is closest to
the `q`-quantile and return it as a [`DatasetSplit`](@ref).
"""
function split_set(v::M, ssplt::SectionSplit; q=0.1) where M <: Union{BitMatrix, AbstractMatrix}
    rmses = ssplt.rmse_v_comb .* ssplt.rmse_vv_comb;
    i = argmin(abs.(rmses .- quantile(rmses, q)))

    train, test = section_to_sets(i, v, ssplt)
    train_inds = vcat(ssplt.sec_inds[ssplt.train_comb[i]]...)
    test_inds = vcat(ssplt.sec_inds[ssplt.test_comb[i]]...)
    
    return DatasetSplit(
        train_inds,
        test_inds,
        train,
        test
    )
end
