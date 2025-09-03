"""
    neuron2dscatter(x, y, v; cmap=:seismic, radius=2, range=nothing,
                    edgewidth=0.5, edgecolor=(:black, 0.1))

Scatter plot of neuron positions coloured by value.

`x` and `y` are coordinate vectors and `v` specifies the colour of each
point. The plot can be customised via the provided keyword arguments.
"""
@recipe(Neuron2DScatter) do scene
    Attributes(
        cmap=:seismic,
        radius=2,
        range=nothing,
        edgewidth=0.5,
        edgecolor=(:black, 0.1),
    )
end

function Makie.plot!(n2ds::Neuron2DScatter{<:Tuple{Vector,Vector,Union{Vector,BitVector}}})
    X = n2ds[1]
    Y = n2ds[2]
    V = n2ds[3]

    if isa(n2ds.range.val, Nothing)
        n2ds.range = @lift((minimum($V), maximum($V)))
    end

    n2ds.colormap = n2ds.cmap
    n2ds.colorrange = n2ds.range

    function update_plot(x, y, v)
        scatter!(
            n2ds,
            x, y,
            markersize=n2ds.radius,
            color=v,
            colormap=n2ds.cmap,
            colorrange=n2ds.range,
            strokewidth=n2ds.edgewidth,
            strokecolor=n2ds.edgecolor,
        )
    end
    Makie.Observables.onany(update_plot, X, Y, V)
    update_plot(X, Y, V)

    n2ds
end

"""
    OrthogonalView

Container holding axes for three orthogonal 2‑D projections.
"""
struct OrthogonalView
    grid::GridLayout
    axXY::Axis
    axYZ::Axis
    axXZ::Axis
end

"""
    orthogonal_view_layout(pos, coords; size=200, constrain=:width)

Construct a layout with three orthogonal views of 3‑D coordinates.

`pos` specifies the parent grid position. The layout size is determined by
`size` and either the width or height is constrained depending on
`constrain`.
"""
function orthogonal_view_layout(pos::GridPosition, coords; size=200, constrain::Symbol=:width)
    spanx, spany, spanz = span_dims(coords)
    spanX, spanY = spanx + spanz, spany + spanz
    percx, percy = spanx / spanX, spany / spanY
    perczX, perczY = spanz / spanX, spanz / spanY

    if constrain == :width
        width = Fixed(size)
        height = Fixed(size * spanY / spanX)
    elseif constrain == :height
        width = Fixed(size * spanX / spanY)
        height = Fixed(size)
    end

    ga = GridLayout(
        pos,
        2, 2,
        colsizes=[Relative(percx), Relative(perczX)],
        rowsizes=[Relative(percy), Relative(perczY)],
        default_rowgap=Fixed(1),
        default_colgap=Fixed(1),
        width=width,
        height=height,
    )

    axXY = Axis(ga[1, 1])
    axYZ = Axis(ga[1, 2])
    axXZ = Axis(ga[2, 1])
    linkyaxes!(axXY, axYZ)
    linkxaxes!(axXY, axXZ)

    return OrthogonalView(ga, axXY, axYZ, axXZ)
end

"""
    neuronorthoscatter(OV, coords, vals; kwargs...)

Plot three orthogonal projections of neuronal coordinates coloured by
`vals`.

`OV` is an [`OrthogonalView`](@ref) returned by
[`orthogonal_view_layout`](@ref).
"""
function neuronorthoscatter(OV::OrthogonalView, coords, vals; cmap=:seismic, range=nothing, edgewidth=0.5, edgecolor=(:black, 0.1))
    neuron2dscatter!(OV.axXY, coords[:, 1], coords[:, 2], vals, cmap=cmap, range=range, edgewidth=edgewidth, edgecolor=edgecolor)
    neuron2dscatter!(OV.axYZ, coords[:, 3], coords[:, 2], vals, cmap=cmap, range=range, edgewidth=edgewidth, edgecolor=edgecolor)
    h = neuron2dscatter!(OV.axXZ, coords[:, 1], coords[:, 3], vals, cmap=cmap, range=range, edgewidth=edgewidth, edgecolor=edgecolor)
    return h
end
