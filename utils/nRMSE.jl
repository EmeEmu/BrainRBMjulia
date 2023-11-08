function rmse(y_true::Vector, y_pred::Vector)
    return sqrt.(sum((y_true .- y_pred).^2) / length(y_true))
end
function rmse(y_true::Base.ReshapedArray, y_pred::Base.ReshapedArray)
    return sqrt.(sum((y_true .- y_pred).^2) / length(y_true))
end

function nRMSE(X::AbstractArray, Y::AbstractArray; X_opt::Union{Nothing,AbstractArray}=nothing, Y_opt::Union{Nothing,AbstractArray}=nothing, max_nb=10^8)
    @assert size(Y) == size(X)
    if !isa(X_opt, Nothing)
        @assert size(X_opt) == size(X)
    end
    if !isa(Y_opt, Nothing)
        @assert size(Y_opt) == size(X)
    end
    
    if length(X) > max_nb
        n = round(Int,sqrt(max_nb))
        indsi = randperm(size(X,1))[1:n]
        indsj = randperm(size(X,1))[1:n]
        @views X = vec(X[indsi,indsj])
        @views Y = vec(Y[indsi,indsj])
        if !isa(X_opt, Nothing)
            #X_opt = vec(X_opt)
            @views X_opt = vec(X_opt[indsi,indsj])
        end
        if !isa(Y_opt, Nothing)
            #Y_opt = vec(Y_opt)
            @views Y_opt = vec(Y_opt[indsi,indsj])
        end 
    else
        X = vec(X)
        Y = vec(Y)
        if !isa(X_opt, Nothing)
            X_opt = vec(X_opt)
        end
        if !isa(Y_opt, Nothing)
            Y_opt = vec(Y_opt)
        end 
    end
    
    RMSE_ord = rmse(X,Y)
    
    #X_shuf = shuffle(X)
    #Y_shuf = shuffle(Y)
    X_shuf = X[shuffle(1:length(X))]
    Y_shuf = Y[shuffle(1:length(Y))]
    RMSE_shuf = rmse(X_shuf, Y_shuf)
    
    if isa(X_opt, Nothing) && isa(Y_opt, Nothing)
        RMSE_opt = 0
    elseif isa(X_opt, Nothing) && !isa(Y_opt, Nothing)
        RMSE_opt = rmse(X,Y_opt)
    elseif !isa(X_opt, Nothing) && !isa(Y_opt, Nothing)
        RMSE_opt = rmse(X_opt,Y_opt)
    else
        error("X_opt cannot be given without also giving Y_opt")
    end
    
    return 1 - (RMSE_ord - RMSE_shuf) / (RMSE_opt - RMSE_shuf)
end

function nRMSE_from_moments(M::MomentsAggregate)
    stats = ["<v>","<h>","<vh>","<vv> - <v><v>","<hh> - <h><h>"]
    d = Dict()
    for a in stats
        d[a] = nRMSE(M.valid[a], M.gen[a], Y_opt=M.train[a])
    end
    return d
end
function nRMSE_from_moments(M::SimpleMomentsAggregate)
    stats = ["<v>","<h>","<vh>","<vv> - <v><v>","<hh> - <h><h>"]
    d = Dict()
    for a in stats
        d[a] = nRMSE(M.data[a], M.gen[a])
    end
    return d
end

function nRMSEs_Lp(nrmses::Dict, p::Int=1; max::Bool=false)
    if max
        v = ones(length(nrmses))
    else
        v = clamp.(values(nrmses), 0, 1)
    end
    return sum(v.^p).^(1/p)
end
function nRMSEs_Lp(nrmses::Vector{Dict{Any, Any}}, p::Int=1; max::Bool=false)
    if max
        return nRMSEs_Lp(nrmses[1], p; max=true)
    else
        return [nRMSEs_Lp(ns, p) for ns in nrmses]
    end
end
function nRMSEs_L4(nrmses::Union{Dict,Vector{Dict{Any, Any}}} ; max::Bool=false)
    return nRMSEs_Lp(nrmses, 4; max=max)
end
