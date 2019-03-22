#
# Coherency
#

struct Coherency <: PairwiseStatistic; end
Base.eltype(::Coherency, X::AbstractArray{Complex{T}}) where {T<:Real} = Complex{T}

# Single input matrix
allocwork(::Coherency, X::AbstractVecOrMat{Complex{T}}) where {T<:Real} = nothing
computestat!(::Coherency, out::AbstractMatrix{Complex{T}}, work::Void,
             X::AbstractVecOrMat{Complex{T}}) where {T<:Real} = 
    cov2coh!(out, Ac_mul_A!(out, X))

# Two input matrices
allocwork(::Coherency, X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}}) where {T<:Real} =
    (cov2coh_work(X), cov2coh_work(Y))
function computestat!(::Coherency, out::AbstractMatrix{Complex{T}},
             work::Tuple{Array{T}, Array{T}},
             X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}}) where T<:Real
    cov2coh!(out, X, Y, work[1], work[2], Ac_mul_B!(out, X, Y))
end

#
# Coherence (as the absolute value of the correlation matrix)
#

struct Coherence <: PairwiseStatistic; end
Base.eltype(::Coherence, X::AbstractArray{Complex{T}}) where {T<:Real} = T

# Single input matrix
allocwork(::Coherence, X::AbstractVecOrMat{Complex{T}}) where {T<:Real} =
    Array(Complex{T}, nchannels(X), nchannels(X))
computestat!(::Coherence, out::AbstractMatrix{T},
             work::Matrix{Complex{T}},
             X::AbstractVecOrMat{Complex{T}}) where {T<:Real} =
    cov2coh!(out, Ac_mul_A!(work, X), Base.AbsFun())

# Two input matrices
allocwork(::Coherence, X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}}) where {T<:Real} =
    (Array(Complex{T}, nchannels(X), nchannels(Y)), cov2coh_work(X), cov2coh_work(Y))
computestat!(::Coherence, out::AbstractMatrix{T},
             work::Tuple{Matrix{Complex{T}}, Array{T}, Array{T}},
             X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}}) where {T<:Real} = 
    cov2coh!(out, X, Y, work[2], work[3], Ac_mul_B!(work[1], X, Y), Base.AbsFun())

#
# Jackknifing for Coherency/Coherence
#
surrogateval(::Coherence, v) = abs(v)
surrogateval(::Coherency, v) = v

# Single input matrix
allocwork(t::Union{AbstractJackknifeSurrogates{Coherency}, AbstractJackknifeSurrogates{Coherence}},
          X::AbstractVecOrMat{Complex{T}}) where {T<:Real} = (allocwork(t.transform, X), Array(T, div(size(X, 1), jnn(t)), size(X, 2)))
accumulator_array(::AbstractJackknifeSurrogates{Coherency}, work::Void, out::AbstractMatrix) = out
accumulator_array(::AbstractJackknifeSurrogates{Coherence}, work::AbstractMatrix, out::AbstractMatrix) = work
function computestat!(t::Union{AbstractJackknifeSurrogates{Coherency}, AbstractJackknifeSurrogates{Coherence}},
                      out::JackknifeSurrogatesOutput,
                      work::Tuple{Union{Matrix{Complex{T}}, Void}, Matrix{T}},
                      X::AbstractVecOrMat{Complex{T}}) where T<:Real
    stat = t.transform
    trueval = out.trueval
    surrogates = out.surrogates
    XXc = accumulator_array(t, work[1], out.trueval)
    jnspec = work[2]
    chkinput(trueval, X)
    ntrials(X) % jnn(t) == 0 || throw(DimensionMismatch("ntrials not evenly divisible by $(jnn(t))"))
    size(out.surrogates, 1) == div(ntrials(X), jnn(t)) || throw(DimensionMismatch("invalid output size"))

    Ac_mul_A!(XXc, X)

    # Compute jackknifed power spectra
    @inbounds for j = 1:size(X, 2)
        ssq = real(XXc[j, j])
        for i = 1:size(surrogates, 1)
            ssqdel = ssq
            for idel = (i-1)*jnn(t)+1:i*jnn(t)
                ssqdel -= abs2(X[idel, j])
            end
            jnspec[i, j] = 1/sqrt(ssqdel)
        end
    end

    # Surrogates
    @inbounds for k = 1:size(X, 2)
        for j = 1:k-1, i = 1:size(surrogates, 1)
            v = XXc[j, k]
            for idel = (i-1)*jnn(t)+1:i*jnn(t)
                v -= conj(X[idel, j])*X[idel, k]
            end
            surrogates[i, j, k] = surrogateval(t.transform, v)*(jnspec[i, j]*jnspec[i, k])
        end
        for i = 1:size(surrogates, 1)
            surrogates[i, k, k] = 1
        end
    end

    # Finish true value
    if isa(t.transform, Coherence)
        cov2coh!(trueval, XXc, Base.AbsFun())
    else
        cov2coh!(trueval, XXc, Base.IdFun())
    end

    out
end

# Two input matrices
allocwork(t::AbstractJackknifeSurrogates{Coherency}, X::AbstractVecOrMat{Complex{T}},
          Y::AbstractVecOrMat{Complex{T}}) where {T<:Real} =
    (nothing, cov2coh_work(X), cov2coh_work(Y), Array(T, div(size(X, 1), jnn(t)), size(X, 2)),
     Array(T, div(size(Y, 1), jnn(t)), size(Y, 2)))
allocwork(t::AbstractJackknifeSurrogates{Coherence}, X::AbstractVecOrMat{Complex{T}},
          Y::AbstractVecOrMat{Complex{T}}) where {T<:Real} =
    (Array(Complex{T}, nchannels(X), nchannels(Y)), cov2coh_work(X), cov2coh_work(Y),
     Array(T, div(size(X, 1), jnn(t)), size(X, 2)), Array(T, div(size(Y, 1), jnn(t)), size(Y, 2)))
function computestat!(t::Union{AbstractJackknifeSurrogates{Coherency}, AbstractJackknifeSurrogates{Coherence}},
                      out::JackknifeSurrogatesOutput,
                      work::Tuple{V, Array{T}, Array{T}, Matrix{T}, Matrix{T}},
                      X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}}) where {T<:Real,V}
    stat = t.transform
    trueval = out.trueval
    surrogates = out.surrogates
    XYc = accumulator_array(t, work[1], out.trueval)
    Xjnspec = work[4]
    Yjnspec = work[5]
    chkinput(trueval, X, Y)
    ntrials(X) % jnn(t) == 0 || throw(DimensionMismatch("ntrials not evenly divisible by $(jnn(t))"))
    size(out.surrogates, 1) == div(ntrials(X), jnn(t)) || throw(DimensionMismatch("invalid output size"))

    Ac_mul_B!(XYc, X, Y)

    # Compute jackknifed power spectra
    @inbounds for (arr, jnspec) in ((X, Xjnspec), (Y, Yjnspec))
        for j = 1:size(arr, 2)
            ssq = zero(T)
            @simd for i = 1:size(arr, 1)
                ssq += abs2(arr[i, j])
            end
            for i = 1:size(surrogates, 1)
                ssqdel = ssq
                for idel = (i-1)*jnn(t)+1:i*jnn(t)
                    ssqdel -= abs2(arr[idel, j])
                end
                jnspec[i, j] = 1/sqrt(ssqdel)
            end
        end
    end

    # Surrogates
    @inbounds for k = 1:size(Y, 2), j = 1:size(X, 2), i = 1:size(surrogates, 1)
        v = XYc[j, k]
        for idel = (i-1)*jnn(t)+1:i*jnn(t)
            v -= conj(X[idel, j])*Y[idel, k]
        end
        # XXX maybe precompute sqrt for each channel and trial?
        surrogates[i, j, k] = surrogateval(t.transform, v)*(Xjnspec[i, j]*Yjnspec[i, k])
    end

    # Finish true value
    if isa(stat, Coherence)
        cov2coh!(trueval, X, Y, work[2], work[3], XYc, Base.AbsFun())
    else
        cov2coh!(trueval, X, Y, work[2], work[3], XYc, Base.IdFun())
    end

    out
end
