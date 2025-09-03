"""
    encode_binary(i::Int, n::Int)

Return the `n`-bit representation of integer `i` as a `BitVector`.

Throws an assertion error if `i` cannot be represented with `n` bits.
"""
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

"""
    encode_binary(seq::Vector{Int}, n::Int)

Encode each integer in `seq` into an `n`-bit column of a `BitMatrix`.
"""
function encode_binary(seq::Vector{Int}, n::Int)
  l = length(seq)
  res = BitMatrix(zeros(Bool, (n, l)))
  for i in 1:l
    res[:, i] .= encode_binary(seq[i], n)
  end
  return res
end

"""
    decode_binary(v::BitVector)

Convert an `n`-bit `BitVector` `v` back to its integer value.
"""
function decode_binary(v::BitVector)
  res = v[1]
  for j in 2:length(v)
    res *= 2
    res += v[j]
  end
  return res
end

"""
    decode_binary(v::BitMatrix)

Decode each column of `v` and return a vector of integers.
"""
function decode_binary(v::BitMatrix)
  l = size(v, 2)
  res = zeros(Int, l)
  for i in 1:l
    res[i] = decode_binary(v[:, i])
  end
  return res
end

"""
    decode_binary(v::BitArray{3})

Decode each matrix slice of a three-dimensional `BitArray`.
The result is a matrix whose columns correspond to the decoded integers
of the slices `v[:, :, k]`.
"""
function decode_binary(v::BitArray{3})
  (l, n) = size(v, 2), size(v, 3)
  res = zeros(Int, l, n)
  for k in 1:n
    res[:, k] .= decode_binary(v[:, :, k])
  end
  return res
end

"""
    Nstates_per_Nbits(N::Int)

Number of distinct states representable by `N` bits, equal to `2^N`.
"""
function Nstates_per_Nbits(N::Int)
  return 2^N
end

"""
    Nbits_per_Nstates(N::Int)

Minimum number of bits required to represent `N` distinct states.
"""
function Nbits_per_Nstates(N::Int)
  return Int(ceil(log2(N)))
end
