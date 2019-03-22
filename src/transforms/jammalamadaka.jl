#
# Jammalamadaka circular correlation
#
# See Jammalamadaka, S. R., & Sengupta, A. (2001). Topics in Circular
# Statistics. World Scientific, p. 176
struct JammalamadakaR{Normalized} <: NormalizedPairwiseStatistic{Normalized} end
JammalamadakaR() = JammalamadakaR{false}()
Base.eltype(::JammalamadakaR, X::AbstractArray{Complex{T}}) where {T<:Real} = T

function sinmeanphasediff!(out, X)
    for j = 1:size(X, 2)
        # Compute mean phase
        m = zero(eltype(X))
        @simd for i = 1:size(X, 1)
            @inbounds m += X[i, j]
        end

        # Normalize to unit length
        m /= abs(m)

        # Compute difference from mean phase
        @simd for i = 1:size(X, 1)
          @inbounds out[i, j] = imag(X[i, j]*conj(m))
      end
    end
    out
end

allocwork(t::JammalamadakaR{true}, X::AbstractVecOrMat{Complex{T}}) where {T<:Real} =
    Array(T, size(X, 1), size(X, 2))
function computestat!(t::JammalamadakaR{true}, out::AbstractMatrix{T},
                      work::Matrix{T}, X::AbstractVecOrMat{Complex{T}}) where T<:Real
    chkinput(out, X)

    # Sins minus mean phases
    workX = sinmeanphasediff!(work, X)

    # Products of phase differences
    At_mul_A!(out, workX)
    cov2coh!(out, out, Base.AbsFun())
end

struct JammalamadakaRWorkXY{T<:Real}
    workX::Matrix{T}
    workY::Matrix{T}
    sumworkX::Matrix{T}
    sumworkY::Matrix{T}
end
function allocwork(t::JammalamadakaR{true}, X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}}) where T<:Real
    JammalamadakaRWorkXY{T}(Array(T, size(X, 1), size(X, 2)),
                            Array(T, size(Y, 1), size(Y, 2)),
                            Array(T, 1, nchannels(X)),
                            Array(T, 1, nchannels(Y)))
end
function computestat!(t::JammalamadakaR{true}, out::AbstractMatrix{T}, work::JammalamadakaRWorkXY{T},
                      X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}}) where T<:Real
    chkinput(out, X, Y)

    # Sins minus mean phases
    workX = sinmeanphasediff!(work.workX, X)
    workY = sinmeanphasediff!(work.workY, Y)

    # Products of phase differences
    At_mul_B!(out, workX, workY)
    cov2coh!(out, workX, workY, work.sumworkX, work.sumworkY, out, Base.AbsFun())
end
