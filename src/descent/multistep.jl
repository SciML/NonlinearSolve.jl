"""
    MultiStepSchemes

This module defines the multistep schemes used in the multistep descent algorithms. The
naming convention follows <name of method><order of convergence>. The name of method is
typically the last names of the authors of the paper that introduced the method.
"""
module MultiStepSchemes

using ConcreteStructs

abstract type AbstractMultiStepScheme end

function Base.show(io::IO, mss::AbstractMultiStepScheme)
    print(io, "MultiStepSchemes.$(string(nameof(typeof(mss)))[3:end])")
end

newton_steps(::Type{T}) where {T <: AbstractMultiStepScheme} = newton_steps(T())

struct __PotraPtak3 <: AbstractMultiStepScheme end
const PotraPtak3 = __PotraPtak3()

newton_steps(::__PotraPtak3) = 2
nintermediates(::__PotraPtak3) = 1

@kwdef @concrete struct __SinghSharma4 <: AbstractMultiStepScheme
    jvp_autodiff = nothing
end
const SinghSharma4 = __SinghSharma4()

newton_steps(::__SinghSharma4) = 4
nintermediates(::__SinghSharma4) = 2

@kwdef @concrete struct __SinghSharma5 <: AbstractMultiStepScheme
    jvp_autodiff = nothing
end
const SinghSharma5 = __SinghSharma5()

newton_steps(::__SinghSharma5) = 4
nintermediates(::__SinghSharma5) = 2

@kwdef @concrete struct __SinghSharma7 <: AbstractMultiStepScheme
    jvp_autodiff = nothing
end
const SinghSharma7 = __SinghSharma7()

newton_steps(::__SinghSharma7) = 6

@generated function display_name(alg::T) where {T <: AbstractMultiStepScheme}
    res = Symbol(first(split(last(split(string(T), ".")), "{"; limit = 2))[3:end])
    return :($(Meta.quot(res)))
end

end

const MSS = MultiStepSchemes

@kwdef @concrete struct GenericMultiStepDescent <: AbstractDescentAlgorithm
    scheme
    linsolve = nothing
    precs = DEFAULT_PRECS
end

Base.show(io::IO, alg::GenericMultiStepDescent) = print(io, "$(alg.scheme)()")

supports_line_search(::GenericMultiStepDescent) = true
supports_trust_region(::GenericMultiStepDescent) = false

@concrete mutable struct GenericMultiStepDescentCache{S} <: AbstractDescentCache
    f
    p
    δu
    δus
    u
    us
    fu
    fus
    internal_cache
    internal_caches
    extra
    extras
    scheme::S
    timer
    nf::Int
end

# FIXME: @internal_caches needs to be updated to support tuples and namedtuples
# @internal_caches GenericMultiStepDescentCache :internal_caches

function __reinit_internal!(cache::GenericMultiStepDescentCache, args...; p = cache.p,
        kwargs...)
    cache.nf = 0
    cache.p = p
    reset_timer!(cache.timer)
end

function __internal_multistep_caches(
        scheme::Union{MSS.__PotraPtak3, MSS.__SinghSharma4, MSS.__SinghSharma5},
        alg::GenericMultiStepDescent, prob, args...;
        shared::Val{N} = Val(1), kwargs...) where {N}
    internal_descent = NewtonDescent(; alg.linsolve, alg.precs)
    return @shared_caches N __internal_init(
        prob, internal_descent, args...; kwargs..., shared = Val(2))
end

__extras_cache(::MSS.AbstractMultiStepScheme, args...; kwargs...) = nothing, nothing

function __internal_init(prob::Union{NonlinearProblem, NonlinearLeastSquaresProblem},
        alg::GenericMultiStepDescent, J, fu, u; shared::Val{N} = Val(1),
        pre_inverted::Val{INV} = False, linsolve_kwargs = (;),
        abstol = nothing, reltol = nothing, timer = get_timer_output(),
        kwargs...) where {INV, N}
    δu, δus = @shared_caches N (@bb δu = similar(u))
    fu_cache, fus_cache = @shared_caches N (ntuple(MSS.nintermediates(alg.scheme)) do i
        @bb xx = similar(fu)
    end)
    u_cache, us_cache = @shared_caches N (ntuple(MSS.nintermediates(alg.scheme)) do i
        @bb xx = similar(u)
    end)
    internal_cache, internal_caches = __internal_multistep_caches(
        alg.scheme, alg, prob, J, fu, u; shared, pre_inverted, linsolve_kwargs,
        abstol, reltol, timer, kwargs...)
    extra, extras = __extras_cache(
        alg.scheme, alg, prob, J, fu, u; shared, pre_inverted, linsolve_kwargs,
        abstol, reltol, timer, kwargs...)
    return GenericMultiStepDescentCache(
        prob.f, prob.p, δu, δus, u_cache, us_cache, fu_cache, fus_cache,
        internal_cache, internal_caches, extra, extras, alg.scheme, timer, 0)
end

function __internal_solve!(cache::GenericMultiStepDescentCache{MSS.__PotraPtak3, INV}, J,
        fu, u, idx::Val = Val(1); skip_solve::Bool = false, new_jacobian::Bool = true,
        kwargs...) where {INV}
    δu = get_du(cache, idx)
    skip_solve && return DescentResult(; δu)

    (y,) = get_internal_cache(cache, Val(:u), idx)
    (fy,) = get_internal_cache(cache, Val(:fu), idx)
    internal_cache = get_internal_cache(cache, Val(:internal_cache), idx)

    @static_timeit cache.timer "descent step" begin
        result_1 = __internal_solve!(
            internal_cache, J, fu, u, Val(1); new_jacobian, kwargs...)
        δx = result_1.δu

        @bb @. y = u + δx
        fy = evaluate_f!!(cache.f, fy, y, cache.p)
        cache.nf += 1

        result_2 = __internal_solve!(
            internal_cache, J, fy, y, Val(2); kwargs...)
        δy = result_2.δu

        @bb @. δu = δx + δy
    end

    set_du!(cache, δu, idx)
    set_internal_cache!(cache, (y,), Val(:u), idx)
    set_internal_cache!(cache, (fy,), Val(:fu), idx)
    set_internal_cache!(cache, internal_cache, Val(:internal_cache), idx)
    return DescentResult(; δu)
end
