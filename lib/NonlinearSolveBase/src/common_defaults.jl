UNITLESS_ABS2(x::Number) = abs2(x)
function UNITLESS_ABS2(x::AbstractArray)
    return mapreduce(UNITLESS_ABS2, Utils.abs2_and_sum, x; init = Utils.zero_init(x))
end
function UNITLESS_ABS2(x::Union{AbstractVectorOfArray, ArrayPartition})
    return mapreduce(
        UNITLESS_ABS2, Utils.abs2_and_sum,
        Utils.get_internal_array(x); init = Utils.zero_init(x)
    )
end

NAN_CHECK(x::Number) = isnan(x)
NAN_CHECK(x::Enum) = false
NAN_CHECK(x::AbstractArray) = any(NAN_CHECK, x)
function NAN_CHECK(x::Union{AbstractVectorOfArray, ArrayPartition})
    return any(NAN_CHECK, Utils.get_internal_array(x))
end

# `@fastmath` must not be used in these norms. Its `nnan`/`ninf` LLVM flags let the
# optimizer constant-fold `isfinite`/`isnan` guards applied to the result away entirely
# (`termination_conditions.jl`'s protective break compiled to `ret i8 0`), so a diverged
# solve was reported as converged. `@simd` already supplies `reassoc`+`contract`, which is
# what vectorizes the reduction, so dropping `@fastmath` keeps the vectorized loop.
L2_NORM(u::Union{AbstractFloat, Complex}) = abs(u)
L2_NORM(u::Number) = sqrt(UNITLESS_ABS2(u))
function L2_NORM(u::Array{<:Union{AbstractFloat, Complex}})
    if Utils.fast_scalar_indexing(u)
        x = zero(eltype(u))
        @simd for i in eachindex(u)
            @inbounds x += abs2(u[i])
        end
        return sqrt(real(x))
    end
    return sqrt(UNITLESS_ABS2(u))
end
function L2_NORM(u::StaticArray{<:Union{AbstractFloat, Complex}})
    return sqrt(real(sum(abs2, u)))
end
L2_NORM(u) = norm(u, 2)

Linf_NORM(u::Union{AbstractFloat, Complex}) = abs(u)
Linf_NORM(u) = maximum(abs, u)

get_tolerance(η, ::Type{T}) where {T} = Utils.convert_real(T, η)
function get_tolerance(::Nothing, ::Type{T}) where {T}
    η = real(oneunit(T)) * (eps(real(one(T)))^(4 // 5))
    return get_tolerance(η, T)
end
function get_tolerance(::Nothing, ::Type{Float64})
    # trimming hangs up on the literal_pow to rational numbers here
    η = real(oneunit(Float64)) * 3.0e-13
    return get_tolerance(η, Float64)
end

get_tolerance(_, η, ::Type{T}) where {T} = get_tolerance(η, T)
function get_tolerance(::Union{StaticArray, Number}, ::Nothing, ::Type{T}) where {T}
    # Rational numbers can throw an error if used inside GPU Kernels
    return T(real(oneunit(T)) * (eps(real(one(T)))^(real(T)(0.8))))
end
