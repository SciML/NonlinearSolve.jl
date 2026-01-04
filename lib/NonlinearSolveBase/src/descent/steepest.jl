"""
    SteepestDescent(; linsolve = nothing)

Compute the descent direction as ``δu = -Jᵀfu``. The linear solver and preconditioner are
only used if `J` is provided in the inverted form.

See also [`Dogleg`](@ref), [`NewtonDescent`](@ref), [`DampedNewtonDescent`](@ref).
"""
@kwdef @concrete struct SteepestDescent <: AbstractDescentDirection
    linsolve = nothing
end

supports_line_search(::SteepestDescent) = true

@concrete mutable struct SteepestDescentCache <: AbstractDescentCache
    δu
    δus
    lincache
    timer
    preinverted_jacobian <: Union{Val{false}, Val{true}}
end

@internal_caches SteepestDescentCache :lincache

function InternalAPI.init(
        prob::AbstractNonlinearProblem, alg::SteepestDescent, J, fu, u;
        stats, shared = Val(1), pre_inverted::Val = Val(false), linsolve_kwargs = (;),
        abstol = nothing, reltol = nothing,
        timer = get_timer_output(),
        kwargs...
    )
    if Utils.unwrap_val(pre_inverted)
        @assert length(fu) == length(u) "Non-Square Jacobian Inverse doesn't make sense."
    end
    @bb δu = similar(u)
    δus = Utils.unwrap_val(shared) ≤ 1 ? nothing : map(2:Utils.unwrap_val(shared)) do i
            @bb δu_ = similar(u)
    end
    if Utils.unwrap_val(pre_inverted)

        if haskey(kwargs, :verbose)
            linsolve_kwargs = merge(
                (verbose = kwargs[:verbose].linear_verbosity,), linsolve_kwargs
            )
        end

        lincache = construct_linear_solver(
            alg, alg.linsolve, transpose(J), Utils.safe_vec(fu), Utils.safe_vec(u);
            stats, abstol, reltol, linsolve_kwargs...
        )
    else
        lincache = nothing
    end
    return SteepestDescentCache(δu, δus, lincache, timer, pre_inverted)
end

function InternalAPI.solve!(
        cache::SteepestDescentCache, J, fu, u, idx::Val = Val(1);
        new_jacobian::Bool = true, kwargs...
    )
    δu = SciMLBase.get_du(cache, idx)
    if Utils.unwrap_val(cache.preinverted_jacobian)
        A = J === nothing ? nothing : transpose(J)
        linres = cache.lincache(;
            A, b = Utils.safe_vec(fu), kwargs..., linu = Utils.safe_vec(δu),
            reuse_A_if_factorization = !new_jacobian || idx !== Val(1)
        )
        δu = Utils.restructure(SciMLBase.get_du(cache, idx), linres.u)
        if !linres.success
            set_du!(cache, δu, idx)
            return DescentResult(; δu, success = false, linsolve_success = false)
        end
    else
        @assert J !== nothing "`J` must be provided when `preinverted_jacobian = Val(false)`."
        @bb δu = transpose(J) × vec(fu)
    end
    @bb @. δu *= -1
    set_du!(cache, δu, idx)
    return DescentResult(; δu)
end
