"""
    rbmdiagram(v, h, W; vcmap=:Greens, hcmap=:plasma, wcmap=cmap_aseismic(),
               vrange=(0,1), hrange=(-1,1), wrange=(-1,1), orientation=:h,
               show_v=true, kwargs...)

Visualise visible and hidden unit activations of an RBM together with its
weights.

`v` and `h` are vectors of visible and hidden values while `W` is the
weight matrix. Colours and layout can be customised via keyword arguments.
"""
@recipe(RBMDiagram) do scene
    Attributes(
        vcmap=:Greens,
        hcmap=:plasma,
        wcmap=cmap_aseismic(),
        vrange=(0,1),
        hrange=(-1,+1),
        wrange=(-1,+1),
        vnode_size=20,
        hnode_size=20,
        layer_gap=2,
        vstep=1,
        hstep=1,
        vedgewidth=1,
        hedgewidth=1,
        orientation=:h,
        offset=0,
        show_v=true,
    )
end

function Makie.plot!(rd::RBMDiagram{<:Tuple{Union{Vector,BitVector}, Union{Vector,BitVector}, Matrix, }})
    V, H, W = rd[1], rd[2], rd[3]
    N, M = length(V.val), length(H.val)
    
    if rd.orientation.val == :h
        Xv = centered_range(N, rd.vstep.val) ; Yv = Xv.*0 .+ rd.offset.val
        Xh = centered_range(M, rd.hstep.val) ; Yh = Xh.*0 .+ rd.layer_gap.val .+ rd.offset.val
    elseif rd.orientation.val == :v
        Yv = centered_range(N, rd.vstep.val) ; Xv = Yv.*0 .+ rd.offset.val
        Yh = centered_range(M, rd.hstep.val) ; Xh = Yh.*0 .+ rd.layer_gap.val .+ rd.offset.val
    end
    
    Xw,Yw = Float32[], Float32[]
    for j in 1:1:M
        for i in 1:1:N
            push!(Xw, Xv[i])
            push!(Xw, Xh[j])
            push!(Yw, Yv[i])
            push!(Yw, Yh[j])
        end
    end
    
    function update_plot(v,xv,yv,h,xh,yh,w,xw,yw)
        c = @lift([($w...)...])
        linesegments!(rd,
            xw, yw, 
            color=c, 
            colormap=rd.wcmap,
            colorrange=rd.wrange,
        )
        if rd.show_v.val
            scatter!(rd, 
                xv, yv, 
                markersize=rd.vnode_size, 
                color=v, 
                colormap=rd.vcmap,
                colorrange=rd.vrange,
                strokewidth=rd.vedgewidth,
            )
        end
        scatter!(rd, 
            xh, yh, 
            markersize=rd.hnode_size, 
            color=h, 
            colormap=rd.hcmap, 
            colorrange=rd.hrange,
            strokewidth=rd.hedgewidth,
        )
    end
    Makie.Observables.onany(update_plot, V,Xv,Yv,H,Xh,Yh,W,Xw,Yw)
    update_plot(V,Xv,Yv,H,Xh,Yh,W,Xw,Yw)
    
    rd
end
