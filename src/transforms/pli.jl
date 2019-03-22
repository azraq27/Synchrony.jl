#
# Phase lag index
#
# See Stam, C. J., Nolte, G., & Daffertshofer, A. (2007).
# Phase lag index: Assessment of functional connectivity from multi
# channel EEG and MEG with diminished bias from common sources.
# Human Brain Mapping, 28(11), 1178–1193. doi:10.1002/hbm.20346
struct PLI <: PairwiseStatistic; end

finish(::Type{PLI}, x::Complex, n::Int) = abs(x)/n

#
# Unbiased squared phase lag index, weighted phase
#
# See Vinck, M., Oostenveld, R., van Wingerden, M., Battaglia, F., &
# Pennartz, C. M. A. (2011). An improved index of phase-synchronization
# for electrophysiological data in the presence of volume-conduction,
# noise and sample-size bias. NeuroImage, 55(4), 1548–1565.
# doi:10.1016/j.neuroimage.2011.01.055
struct PLI2Unbiased <: PairwiseStatistic; end

accumulator(::Union{Type{PLI}, Type{PLI2Unbiased}}, ::Type{T}) where {T<:Real} = zero(Complex{T})
accumulate(::Union{Type{PLI}, Type{PLI2Unbiased}}, x::Complex{T},
           v1::Complex{T}, v2::Complex{T}) where {T<:Real} = (x + sign(imag(conj(v1)*v2)))
accumulate(::Union{Type{PLI}, Type{PLI2Unbiased}}, x::Complex{T},
           v1::Complex{T}, v2::Complex{T}, weight::Real) where {T<:Real} = (x + sign(imag(conj(v1)*v2))*weight)
finish(::Type{PLI2Unbiased}, x::Complex, n::Int) = abs2(x)/(n*(n-1)) - 1/(n-1)

#
# Weighted phase lag index
#
# See Vinck et al. (2011) as above.
struct WPLI <: PairwiseStatistic; end
struct WPLIAccumulator{T}
    si::Complex{T}   # Sum of imag(conj(v1)*(v2))
    sa::T            # Sum of abs(imag(conj(v1)*(v2)))
end

accumulator(::Type{WPLI}, ::Type{T}) where {T<:Real} =
    WPLIAccumulator{T}(zero(Complex{T}), zero(T))
function accumulate(::Type{WPLI}, x::WPLIAccumulator{T},
                    v1::Complex{T}, v2::Complex{T}) where T<:Real
    z = imag(conj(v1)*v2)
    WPLIAccumulator(x.si + z, x.sa + abs(z))
end
function accumulate(::Type{WPLI}, x::WPLIAccumulator{T},
                    v1::Complex{T}, v2::Complex{T}, weight::Real) where T<:Real
    z = imag(conj(v1)*v2)
    WPLIAccumulator(x.si + z*weight, x.sa + abs(z*weight))
end
finish(::Type{WPLI}, x::WPLIAccumulator, n::Int) = abs(x.si)/x.sa

#
# Debiased (i.e. still somewhat biased) WPLI^2
#
# See Vinck et al. (2011) as above.
struct WPLI2Debiased <: PairwiseStatistic; end
struct WPLI2DebiasedAccumulator{T}
    si::Complex{T}   # Sum of imag(conj(v1)*(v2))
    sa::T            # Sum of abs(imag(conj(v1)*(v2)))
    sa2::T           # Sum of abs2(imag(conj(v1)*(v2)))
end

accumulator(::Type{WPLI2Debiased}, ::Type{T}) where {T<:Real} =
    WPLI2DebiasedAccumulator{T}(zero(Complex{T}), zero(T), zero(T))
function accumulate(::Type{WPLI2Debiased}, x::WPLI2DebiasedAccumulator{T},
                    v1::Complex{T}, v2::Complex{T}) where T<:Real
    z = imag(conj(v1)*v2)
    WPLI2DebiasedAccumulator(x.si + z, x.sa + abs(z), x.sa2 + abs2(z))
end
function accumulate(::Type{WPLI2Debiased}, x::WPLI2DebiasedAccumulator{T},
                    v1::Complex{T}, v2::Complex{T}, weight::Real) where T<:Real
    z = imag(conj(v1)*v2)
    WPLI2DebiasedAccumulator(x.si + z*weight, x.sa + abs(z*weight), x.sa2 + abs2(z)*weight)
end
finish(::Type{WPLI2Debiased}, x::WPLI2DebiasedAccumulator,
       n::Int) = (abs2(x.si) - x.sa2)/(abs2(x.sa) - x.sa2)

#
# Functions applicable to all phase lag-style metrics
#
const PLStat = Union{PLI, PLI2Unbiased, WPLI, WPLI2Debiased}
Base.eltype(::PLStat, X::AbstractArray{Complex{T}}) where {T<:Real} = T
diagval(::Type{T}) where {T<:PLStat} = 0

allocwork(::PLStat, X::AbstractVecOrMat{T}, Y::AbstractVecOrMat{T}=X) where {T<:Complex} =
    nothing

# Single input matrix
function computestat!(t::S, out::AbstractMatrix{T}, work::Nothing,
                      X::AbstractVecOrMat{Complex{T}}) where {S<:PLStat, T<:Real}
    chkinput(out, X)
    for k = 1:size(X, 2), j = 1:k
        v = accumulator(S, T)
        @simd for i = 1:size(X, 1)
            @inbounds v = accumulate(S, v, X[i, j], X[i, k])
        end
        out[j, k] = finish(S, v, size(X, 1))
    end
    out
end

# Two input matrices
function computestat!(t::S, out::AbstractMatrix{T}, work::Nothing,
                      X::AbstractVecOrMat{Complex{T}},
                      Y::AbstractVecOrMat{Complex{T}}) where {S<:PLStat, T<:Real}
    chkinput(out, X, Y)
    for k = 1:size(Y, 2), j = 1:size(X, 2)
        v = accumulator(S, T)
        @simd for i = 1:size(X, 1)
            @inbounds v = accumulate(S, v, X[i, j], Y[i, k])
        end
        out[j, k] = finish(S, v, size(X, 1))
    end
    out
end
