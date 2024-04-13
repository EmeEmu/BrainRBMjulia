function reorder_hus(rbm::StandardizedRBM, order::Vector{Int})
    new = deepcopy(rbm)
    new.w .= rbm.w[:,order]
    new.hidden.par .= rbm.hidden.par[:,order]
    new.offset_h .= rbm.offset_h[order]
    new.scale_h .= rbm.scale_h[order]
    return new
end

function reorder_hus!(rbm::StandardizedRBM, v::AbstractArray)
    hh = mean_h_from_v(rbm, v);
    C = cor(hh')
    HCLUST = hclust(1 .-C, linkage=:ward, branchorder=:optimal)
    order = HCLUST.order
    rbm.w .= rbm.w[:,order]
    rbm.hidden.par .= rbm.hidden.par[:,order]
    rbm.offset_h .= rbm.offset_h[order]
    rbm.scale_h .= rbm.scale_h[order]
    rbm
end

function linear_order(coords::Matrix{Int64}, o::Vector)
    lc = coords[:,o[1]]./maximum(coords[:,o[1]]).*100 + coords[:,o[2]]./maximum(coords[:,o[2]]).*1 + coords[:,o[3]]./maximum(coords[:,o[3]]).*0.01
    sort_vi = sortperm(lc)
    return sort_vi
end

function linear_order(vox::VoxelGrid, o::Vector)
    return linear_order(getindex.(vox.goods, [1 2 3]), o)
end

function reorder_states_corr(X::AbstractMatrix)
    C = cor(X)
    C[isnan.(C)] .= 0
    HCLUST = hclust(1 .- C, linkage=:ward, branchorder=:optimal)
    sorder = HCLUST.order;
    return sorder
end
