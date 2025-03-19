function encode_binary(i::Int, n::Int)
  @assert i <= (2^n - 1) "can't uncode `$(i)` with `$(n)` bits !"
  res = BitVector(zeros(Bool, n))
  for j in 1:n
    res[n-j+1] = i % 2
    i = i ÷ 2
  end
  return res
  # digits(3, base=2, pad = 5)
end

function encode_binary(seq::Vector{Int}, n::Int)
  l = length(seq)
  res = BitMatrix(zeros(Bool, (n, l)))
  for i in 1:l
    res[:, i] .= encode_binary(seq[i], n)
  end
  return res
end

function decode_binary(v::BitVector)
  res = v[1]
  for j in 2:length(v)
    res *= 2
    res += v[j]
  end
  return res
end

function decode_binary(v::BitMatrix)
  l = size(v, 2)
  res = zeros(Int, l)
  for i in 1:l
    res[i] = decode_binary(v[:, i])
  end
  return res
end

function decode_binary(v::BitArray{3})
  (l, n) = size(v, 2), size(v, 3)
  res = zeros(Int, l, n)
  for k in 1:n
    res[:, k] .= decode_binary(v[:, :, k])
  end
  return res
end

function Nstates_per_Nbits(N::Int)
  return 2^N
end
function Nbits_per_Nstates(N::Int)
  return Int(ceil(log2(N)))
end
