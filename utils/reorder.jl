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