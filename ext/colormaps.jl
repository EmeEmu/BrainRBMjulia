"""
    colorscheme_alpha(cscheme, alpha=0.5; ncolors=100)

Apply a fixed transparency `alpha` to all colours of `cscheme`.

The resulting colour scheme has `ncolors` samples evenly spaced between 0
and 1.
"""
function colorscheme_alpha(cscheme::ColorScheme, alpha=0.5; ncolors=100)
    return ColorScheme([RGBA(get(cscheme, k), alpha) for k in range(0, 1, length=ncolors)])
end

"""
    colorscheme_whiteTOalpha(cscheme, pow=10; ncolors=100)

Fade opaque colours to transparent white depending on intensity.

Colours closer to white become more transparent following a power-law
with exponent `pow`.
"""
function colorscheme_whiteTOalpha(cscheme::ColorScheme, pow=10; ncolors=100)
    res = colorscheme_alpha(cscheme, 1; ncolors=ncolors)
    res2 = RGBA{Float64}[]
    for k in 1:1:ncolors
        a = (res[k].r + res[k].g + res[k].b) / 3
        push!(res2, RGBA(res[k].r, res[k].g, res[k].b, 1 - a^pow))
    end
    return res2
end

"""
    colorscheme_alpha_sigmoid(cscheme, λ=0.075, μ=0.5, top=0.75; ncolors=100)

Modulate colour transparency using a sigmoid curve.

The opacity increases smoothly from 0 to `top` around the midpoint `μ`
with slope controlled by `λ`.
"""
function colorscheme_alpha_sigmoid(cscheme::ColorScheme, λ=0.075, μ=0.5, top=0.75; ncolors=100)
    function sigmoid(x, λ=0.1, μ=0.5)
        return 1 ./ (1 .+ exp.(-(x .- μ) ./ λ))
    end
    return ColorScheme([RGBA(get(cscheme, k), sigmoid(k, λ, μ) * top) for k in range(0, 1, length=ncolors)])
end

"""
    colorscheme_whiteTOalpha_sigmoid(cscheme, pow=1, λ=0.075, μ=0.5, top=0.75)

Fade colours towards transparency based on brightness and a sigmoid curve.

Colour intensity is raised to `pow` before being passed through a sigmoid
with parameters `λ` and `μ`. The result scales between 0 and `top`.
"""
function colorscheme_whiteTOalpha_sigmoid(cscheme::ColorScheme, pow=1, λ=0.075, μ=0.5, top=0.75)
    function sigmoid(x)
        return 1 ./ (1 .+ exp.(-(x .- μ) ./ λ))
    end
    res = RGBA{Float64}[]
    for k in 1:1:length(cscheme)
        a = (cscheme[k].r + cscheme[k].g + cscheme[k].b) / 3
        a = a^pow
        a = (1 - sigmoid(a)) * top
        push!(res, RGBA(cscheme[k].r, cscheme[k].g, cscheme[k].b, a))
    end
    return res
end

"""
    cmap_aseismic()

Seismic colour scheme that fades to transparency towards white.
"""
cmap_aseismic() = colorscheme_whiteTOalpha(ColorSchemes.seismic)

"""
    cmap_hardseismic()

High-contrast seismic colour scheme with sigmoid transparency.
"""
cmap_hardseismic() = colorscheme_whiteTOalpha_sigmoid(ColorSchemes.seismic, 2, 0.01, 0.5, 1)

"""
    cmap_dff()

Colour scheme suited for ΔF/F fluorescence signals.
"""
cmap_dff() = colorscheme_alpha_sigmoid(ColorSchemes.linear_kgy_5_95_c69_n256)

"""
    cmap_Gbin()

Binary colour map with transparent background and green foreground.
"""
cmap_Gbin() = ColorScheme([RGBA(1, 1, 1, 0), RGBA(0, 1, 0, 1)])

"""
    cmap_ainferno()

Inferno colour scheme with sigmoid transparency.
"""
cmap_ainferno() = colorscheme_alpha_sigmoid(ColorSchemes.inferno, 0.01, 0.1, 0.90)
