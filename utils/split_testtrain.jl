struct DatasetSplit
    train_inds::Vector{Int}
    valid_inds::Vector{Int}
    train::Union{Matrix, BitMatrix}
    valid::Union{Matrix, BitMatrix}
end
    
function split_set(X::Union{Matrix, BitMatrix}; p_train=0.75)
    i_train = Int(floor(p_train*size(X,2)));
    inds = range(1,size(X,2)) |> collect;
    inds = shuffle(inds);
    train_inds = inds[begin:i_train];
    valid_inds = inds[i_train:end];
    train = X[:,train_inds];
    valid = X[:,valid_inds];
    return DatasetSplit(
        train_inds, 
        valid_inds, 
        train, 
        valid
    )
end