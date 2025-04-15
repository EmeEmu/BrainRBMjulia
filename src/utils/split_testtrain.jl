struct DatasetSplit
  train_inds::Vector{Int}
  valid_inds::Vector{Int}
  train::Union{Matrix,BitMatrix}
  valid::Union{Matrix,BitMatrix}
end


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

mutable struct SectionSplit
  N::Int # number of sections
  ftrain::AbstractFloat # fraction of train set
  ftest::AbstractFloat # fraction of test set
  sec_inds::Vector{Vector{Int64}} # time indices of section
  train_comb::Vector{Vector{Int64}} # all possible training sets (by sections)
  test_comb::Vector{Vector{Int64}} # all possible testing sets (by sections)
  rmse_v_comb::Vector{Float64} # <v> moment between train and test sets for all combinations
  rmse_vv_comb::Vector{Float64} # <vv> moment between train and test sets for all combinations

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

  function SectionSplit(v::M, ftrain::AbstractFloat; N::Int=10, N_vv::Int=5000) where {M<:Union{BitMatrix,AbstractMatrix}}
    ssplt = SectionSplit(size(v, 2), ftrain; N)
    rmse_v, rmse_vv = sections_to_rmses(v, ssplt; N_vv)
    ssplt.rmse_v_comb = rmse_v
    ssplt.rmse_vv_comb = rmse_vv
    return ssplt
  end
end


function section_moments(split1::M, split2::M; N_vv::Int=100) where {M<:Union{BitMatrix,AbstractMatrix}}
  m_v_1 = mean(split1, dims=2)[:, 1]
  m_v_2 = mean(split2, dims=2)[:, 1]
  ni = shuffle(1:size(split1, 1))[1:N_vv]
  m_vv_1 = cov(split1[ni, :], dims=2, corrected=false)
  m_vv_2 = cov(split2[ni, :], dims=2, corrected=false)
  return m_v_1, m_v_2, m_vv_1, m_vv_2
end

function section_to_sets(i::Int, v::M, ssplt::SectionSplit) where {M<:Union{BitMatrix,AbstractMatrix}}
  train_inds = vcat(ssplt.sec_inds[ssplt.train_comb[i]]...)
  test_inds = vcat(ssplt.sec_inds[ssplt.test_comb[i]]...)
  train = v[:, train_inds]
  test = v[:, test_inds]
  return train, test
end

function section_to_moments(i::Int, v::M, ssplt::SectionSplit; N_vv::Int=100) where {M<:Union{BitMatrix,AbstractMatrix}}
  train, test = section_to_sets(i, v, ssplt)
  return section_moments(train, test; N_vv)
end

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
