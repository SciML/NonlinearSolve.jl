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

alg_steps(::Type{T}) where {T <: AbstractMultiStepScheme} = alg_steps(T())

struct __PotraPtak3 <: AbstractMultiStepScheme end
const PotraPtak3 = __PotraPtak3()

alg_steps(::__PotraPtak3) = 2

@kwdef @concrete struct __SinghSharma4 <: AbstractMultiStepScheme
    vjp_autodiff = nothing
end
const SinghSharma4 = __SinghSharma4()

alg_steps(::__SinghSharma4) = 3

@kwdef @concrete struct __SinghSharma5 <: AbstractMultiStepScheme
    vjp_autodiff = nothing
end
const SinghSharma5 = __SinghSharma5()

alg_steps(::__SinghSharma5) = 3

@kwdef @concrete struct __SinghSharma7 <: AbstractMultiStepScheme
    vjp_autodiff = nothing
end
const SinghSharma7 = __SinghSharma7()

alg_steps(::__SinghSharma7) = 4

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

supports_line_search(::GenericMultiStepDescent) = false
supports_trust_region(::GenericMultiStepDescent) = false

@concrete mutable struct GenericMultiStepDescentCache{S, INV} <: AbstractDescentCache
    f
    p
    δu
    δus
    extras
    scheme::S
    lincache
    timer
    nf::Int
end

@internal_caches GenericMultiStepDescentCache :lincache

function __reinit_internal!(cache::GenericMultiStepDescentCache, args...; p = cache.p,
        kwargs...)
    cache.nf = 0
    cache.p = p
end

function __δu_caches(scheme::MSS.__PotraPtak3, fu, u, ::Val{N}) where {N}
    caches = ntuple(N) do i
        @bb δu = similar(u)
        @bb y = similar(u)
        @bb fy = similar(fu)
        @bb δy = similar(u)
        @bb u_new = similar(u)
        (δu, δy, fy, y, u_new)
    end
    return first(caches), (N ≤ 1 ? nothing : caches[2:end])
end

function __internal_init(prob::NonlinearProblem, alg::GenericMultiStepDescent, J, fu, u;
        shared::Val{N} = Val(1), pre_inverted::Val{INV} = False, linsolve_kwargs = (;),
        abstol = nothing, reltol = nothing, timer = get_timer_output(),
        kwargs...) where {INV, N}
    δu, δus = __δu_caches(alg.scheme, fu, u, shared)
    INV && return GenericMultiStepDescentCache{true}(prob.f, prob.p, δu, δus,
        alg.scheme, nothing, timer, 0)
    lincache = LinearSolverCache(alg, alg.linsolve, J, _vec(fu), _vec(u); abstol, reltol,
        linsolve_kwargs...)
    return GenericMultiStepDescentCache{false}(prob.f, prob.p, δu, δus, alg.scheme,
        lincache, timer, 0)
end

function __internal_init(prob::NonlinearLeastSquaresProblem, alg::GenericMultiStepDescent,
        J, fu, u; kwargs...)
    error("Multi-Step Descent Algorithms for NLLS are not implemented yet.")
end

function __internal_solve!(cache::GenericMultiStepDescentCache{MSS.__PotraPtak3, INV}, J,
        fu, u, idx::Val = Val(1); skip_solve::Bool = false, new_jacobian::Bool = true,
        kwargs...) where {INV}
    (u_new, δy, fy, y, δu) = get_du(cache, idx)
    skip_solve && return DescentResult(; u = u_new)

    @static_timeit cache.timer "linear solve" begin
        @static_timeit cache.timer "solve and step 1" begin
            if INV
                J !== nothing && @bb(δu=J × _vec(fu))
            else
                δu = cache.lincache(; A = J, b = _vec(fu), kwargs..., linu = _vec(δu),
                    du = _vec(δu),
                    reuse_A_if_factorization = !new_jacobian || (idx !== Val(1)))
                δu = _restructure(u, δu)
            end
            @bb @. y = u - δu
        end

        fy = evaluate_f!!(cache.f, fy, y, cache.p)
        cache.nf += 1

        @static_timeit cache.timer "solve and step 2" begin
            if INV
                J !== nothing && @bb(δy=J × _vec(fy))
            else
                δy = cache.lincache(; A = J, b = _vec(fy), kwargs..., linu = _vec(δy),
                    du = _vec(δy), reuse_A_if_factorization = true)
                δy = _restructure(u, δy)
            end
            @bb @. u_new = y - δy
        end
    end

    set_du!(cache, (u_new, δy, fy, y, δu), idx)
    return DescentResult(; u = u_new)
end
