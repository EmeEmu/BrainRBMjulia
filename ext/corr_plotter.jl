@recipe(CorrPlotter) do scene
    Attributes(
        cmap=:seismic,
        order=nothing,
    )
end

function Makie.plot!(cp::CorrPlotter{<:Tuple{AbstractArray}})
    x = cp[1]
    if @lift(size($x,1) == size($x,2)).val
        corr = x
    else
        corr = @lift(cor($x))
    end
    
    if isa(cp.order.val, Vector)
        corr = @lift($corr[cp.order.val,cp.order.val])
    elseif cp.order.val == "hier"
        HCLUST = @lift(hclust(1 .- $corr, linkage=:ward, branchorder=:optimal))
        order = @lift($HCLUST.order)
        corr = @lift($corr[$order,$order])
    end
    
    h = heatmap!(cp, corr, colormap=cp.cmap, colorrange=(-1,+1))
    cp.colormap = h.colormap
    cp.colorrange = h.colorrange
    
    cp
end
      
@recipe(CouplingPlotter) do scene
    Attributes(
        cmap=:seismic,
        order=nothing,
        colorquantile=0.95,
    )
end

function Makie.plot!(cp::CouplingPlotter{<:Tuple{AbstractArray}})
    J = cp[1]
    
    if isa(cp.order.val, Vector)
        J = @lift($J[cp.order.val,cp.order.val])
    elseif cp.order.val == "hier"
        HCLUST = @lift(hclust(1 .- ($J + $J')/2, linkage=:ward, branchorder=:optimal))
        order = @lift($HCLUST.order)
        J = @lift($J[$order,$order])
    end
    
    l = @lift(quantile_range($J, cp.colorquantile.val))
    h = heatmap!(cp, J, colormap=cp.cmap, colorrange=@lift((-$l,+$l)))
    cp.colormap = h.colormap
    cp.colorrange = h.colorrange
    
    cp
end
