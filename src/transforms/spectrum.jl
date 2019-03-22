#
# Power spectrum
#

struct PowerSpectrum <: Statistic; end

Base.eltype(::PowerSpectrum, X::AbstractVecOrMat{Complex{T}}) where {T<:Real} = T
allocwork(::PowerSpectrum, X::AbstractVecOrMat{Complex{T}}) where {T<:Complex} = nothing
allocoutput(::PowerSpectrum, X::AbstractVecOrMat{Complex{T}}) where {T<:Real} =
    Array(T, 1, nchannels(X))

# Single input matrix
computestat!(::PowerSpectrum, out::AbstractMatrix{T}, ::Nothing,
             X::AbstractVecOrMat{Complex{T}}) where {T<:Real} =
    scale!(sumabs2!(out, X), 1/ntrials(X))

#
# Cross spectrum
#

struct CrossSpectrum <: PairwiseStatistic; end
Base.eltype(::CrossSpectrum, X::AbstractArray{Complex{T}}) where {T<:Real} = Complex{T}

# Single input matrix
allocwork(::CrossSpectrum, X::AbstractVecOrMat{T}) where {T<:Complex} = nothing
computestat!(::CrossSpectrum, out::AbstractMatrix{T}, ::Nothing,
             X::AbstractVecOrMat{T}) where {T<:Complex} =
    scale!(Ac_mul_A!(out, X), 1/ntrials(X))

# Two input matrices
allocwork(::CrossSpectrum, X::AbstractVecOrMat{T}, Y::AbstractVecOrMat{T}) where {T<:Complex} = nothing
computestat!(::CrossSpectrum, out::AbstractMatrix{T}, ::Nothing,
             X::AbstractVecOrMat{T}, Y::AbstractVecOrMat{T}) where {T<:Complex} =
    scale!(Ac_mul_B!(out, X, Y), 1/ntrials(X))

accumulator(::Type{CrossSpectrum}, ::Type{T}) where {T<:Real} = zero(Complex{T})
@inline accumulate(::Type{CrossSpectrum}, x::Complex{T},
                   v1::Complex{T}, v2::Complex{T}) where {T<:Real} = (x + conj(v1)*v2)
@inline accumulate(::Type{CrossSpectrum}, x::Complex{T},
                   v1::Complex{T}, v2::Complex{T}, weight::Real) where {T<:Real} = (x + conj(v1)*v2*weight)
finish(::Type{CrossSpectrum}, x::Complex, n::Int) = x/n
