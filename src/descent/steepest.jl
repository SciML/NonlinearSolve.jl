"""
    SteepestDescent(; linsolve = nothing, precs = DEFAULT_PRECS)

Compute the descent direction as ``δu = -Jᵀfu``. The linear solver and preconditioner are
only used if `J` is provided in the inverted form.

See also [`Dogleg`](@ref), [`NewtonDescent`](@ref), [`DampedNewtonDescent`](@ref).
"""
@kwdef @concrete struct SteepestDescent <: AbstractDescentAlgorithm
    linsolve = nothing
    precs = DEFAULT_PRECS
end

function Base.show(io::IO, d::SteepestDescent)
    modifiers = String[]
    d.linsolve !== nothing && push!(modifiers, "linsolve = $(d.linsolve)")
    d.precs !== DEFAULT_PRECS && push!(modifiers, "precs = $(d.precs)")
    print(io, "SteepestDescent($(join(modifiers, ", ")))")
end

supports_line_search(::SteepestDescent) = true

@concrete mutable struct SteepestDescentCache{pre_inverted} <: AbstractDescentCache
    δu
    δus
    lincache
    timer
end

@internal_caches SteepestDescentCache :lincache

@inline function __internal_init(
        prob::AbstractNonlinearProblem, alg::SteepestDescent, J, fu,
        u; stats, shared::Val{N} = Val(1), pre_inverted::Val{INV} = False,
        linsolve_kwargs = (;), abstol = nothing, reltol = nothing,
        timer = get_timer_output(), kwargs...) where {INV, N}
    INV && @assert length(fu)==length(u) "Non-Square Jacobian Inverse doesn't make sense."
    @bb δu = similar(u)
    δus = N ≤ 1 ? nothing : map(2:N) do i
        @bb δu_ = similar(u)
    end
    if INV
        lincache = LinearSolverCache(alg, alg.linsolve, transpose(J), _vec(fu),
            _vec(u); stats, abstol, reltol, linsolve_kwargs...)
    else
        lincache = nothing
    end
    return SteepestDescentCache{INV}(δu, δus, lincache, timer)
end

function __internal_solve!(cache::SteepestDescentCache{INV}, J, fu, u, idx::Val = Val(1);
        new_jacobian::Bool = true, kwargs...) where {INV}
    δu = get_du(cache, idx)
    if INV
        A = J === nothing ? nothing : transpose(J)
        @static_timeit cache.timer "linear solve" begin
            linres = cache.lincache(;
                A, b = _vec(fu), kwargs..., linu = _vec(δu), du = _vec(δu),
                reuse_A_if_factorization = !new_jacobian || idx !== Val(1))
            δu = _restructure(get_du(cache, idx), linres.u)
            if !linres.success
                set_du!(cache, δu, idx)
                return DescentResult(; δu, success = false, linsolve_success = false)
            end
        end
    else
        @assert J!==nothing "`J` must be provided when `pre_inverted = Val(false)`."
        @bb δu = transpose(J) × vec(fu)
    end
    @bb @. δu *= -1
    set_du!(cache, δu, idx)
    return DescentResult(; δu)
end
