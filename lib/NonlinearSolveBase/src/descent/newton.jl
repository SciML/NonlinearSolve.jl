"""
    NewtonDescent(; linsolve = nothing)

Compute the descent direction as ``J δu = -fu``. For non-square Jacobian problems, this is
commonly referred to as the Gauss-Newton Descent.

See also [`Dogleg`](@ref), [`SteepestDescent`](@ref), [`DampedNewtonDescent`](@ref).
"""
@kwdef @concrete struct NewtonDescent <: AbstractDescentDirection
    linsolve = nothing
end

supports_line_search(::NewtonDescent) = true

@concrete mutable struct NewtonDescentCache <: AbstractDescentCache
    δu
    δus
    lincache
    JᵀJ_cache  # For normal form else nothing
    Jᵀfu_cache
    timer
    preinverted_jacobian <: Union{Val{false}, Val{true}}
    normal_form <: Union{Val{false}, Val{true}}
end

@internal_caches NewtonDescentCache :lincache

function InternalAPI.init(
        prob::AbstractNonlinearProblem, alg::NewtonDescent, J, fu, u; stats,
        shared = Val(1), pre_inverted::Val = Val(false), linsolve_kwargs = (;),
        abstol = nothing, reltol = nothing,
        timer = get_timer_output(), kwargs...
)
    @bb δu = similar(u)
    δus = Utils.unwrap_val(shared) ≤ 1 ? nothing : map(2:Utils.unwrap_val(shared)) do i
        @bb δu_ = similar(u)
    end

    if Utils.unwrap_val(pre_inverted)
        lincache = nothing
    else
        lincache = construct_linear_solver(
            alg, alg.linsolve, J, Utils.safe_vec(fu), Utils.safe_vec(u);
            stats, abstol, reltol, linsolve_kwargs...
        )
    end
    return NewtonDescentCache(
        δu, δus, lincache, nothing, nothing, timer, pre_inverted, Val(false)
    )
end

function InternalAPI.init(
        prob::NonlinearLeastSquaresProblem, alg::NewtonDescent, J, fu, u; stats,
        shared = Val(1), pre_inverted::Val = Val(false), linsolve_kwargs = (;),
        abstol = nothing, reltol = nothing,
        timer = get_timer_output(), kwargs...
)
    length(fu) != length(u) &&
        @assert !Utils.unwrap_val(pre_inverted) "Precomputed Inverse for Non-Square Jacobian doesn't make sense."

    @bb δu = similar(u)
    δus = Utils.unwrap_val(shared) ≤ 1 ? nothing : map(2:N) do i
        @bb δu_ = similar(u)
    end
    normal_form = needs_square_A(alg.linsolve, u)
    if normal_form
        JᵀJ = transpose(J) * J
        Jᵀfu = transpose(J) * Utils.safe_vec(fu)
        A, b = Utils.maybe_symmetric(JᵀJ), Jᵀfu
    else
        JᵀJ, Jᵀfu = nothing, nothing
        A, b = J, Utils.safe_vec(fu)
    end

    lincache = construct_linear_solver(
        alg, alg.linsolve, A, b, Utils.safe_vec(u);
        stats, abstol, reltol, linsolve_kwargs...
    )

    return NewtonDescentCache(
        δu, δus, lincache, JᵀJ, Jᵀfu, timer, pre_inverted, Val(normal_form)
    )
end

function InternalAPI.solve!(
        cache::NewtonDescentCache, J, fu, u, idx::Val = Val(1);
        skip_solve::Bool = false, new_jacobian::Bool = true, kwargs...
)
    δu = SciMLBase.get_du(cache, idx)
    skip_solve && return DescentResult(; δu)
    if preinverted_jacobian(cache) && !normal_form(cache)
        @assert J!==nothing "`J` must be provided when `preinverted_jacobian = Val(true)`."
        @bb δu = J × vec(fu)
    else
        if normal_form(cache)
            @assert !preinverted_jacobian(cache)
            if idx === Val(1)
                @bb cache.JᵀJ_cache = transpose(J) × J
            end
            @bb cache.Jᵀfu_cache = transpose(J) × vec(fu)
            @static_timeit cache.timer "linear solve" begin
                linres = cache.lincache(;
                    A = Utils.maybe_symmetric(cache.JᵀJ_cache), b = cache.Jᵀfu_cache,
                    kwargs..., linu = Utils.safe_vec(δu),
                    reuse_A_if_factorization = !new_jacobian || (idx !== Val(1))
                )
            end
        else
            @static_timeit cache.timer "linear solve" begin
                linres = cache.lincache(;
                    A = J, b = Utils.safe_vec(fu),
                    kwargs..., linu = Utils.safe_vec(δu),
                    reuse_A_if_factorization = !new_jacobian || idx !== Val(1)
                )
            end
        end
        @static_timeit cache.timer "linear solve" begin
            δu = Utils.restructure(SciMLBase.get_du(cache, idx), linres.u)
            if !linres.success
                set_du!(cache, δu, idx)
                return DescentResult(; δu, success = false, linsolve_success = false)
            end
        end
    end

    @bb @. δu *= -1
    set_du!(cache, δu, idx)
    return DescentResult(; δu)
end
