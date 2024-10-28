"""
    Dogleg(; linsolve = nothing, precs = nothing)

Switch between Newton's method and the steepest descent method depending on the size of the
trust region. The trust region is specified via keyword argument `trust_region` to
`solve!`.

See also [`SteepestDescent`](@ref), [`NewtonDescent`](@ref), [`DampedNewtonDescent`](@ref).
"""
@concrete struct Dogleg <: AbstractDescentDirection
    newton_descent <: Union{NewtonDescent, DampedNewtonDescent}
    steepest_descent <: SteepestDescent
end

supports_trust_region(::Dogleg) = true
get_linear_solver(alg::Dogleg) = get_linear_solver(alg.newton_descent)

function Dogleg(; linsolve = nothing, precs = nothing, damping = Val(false),
        damping_fn = missing, initial_damping = missing, kwargs...)
    if !Utils.unwrap_val(damping)
        return Dogleg(NewtonDescent(; linsolve, precs), SteepestDescent(; linsolve, precs))
    end
    if damping_fn === missing || initial_damping === missing
        throw(ArgumentError("`damping_fn` and `initial_damping` must be supplied if \
                             `damping = Val(true)`."))
    end
    return Dogleg(
        DampedNewtonDescent(; linsolve, precs, damping_fn, initial_damping),
        SteepestDescent(; linsolve, precs)
    )
end

@concrete mutable struct DoglegCache <: AbstractDescentCache
    δu
    δus
    newton_cache <: Union{NewtonDescentCache, DampedNewtonDescentCache}
    cauchy_cache <: SteepestDescentCache
    internalnorm
    Jᵀδu_cache
    δu_cache_1
    δu_cache_2
    δu_cache_mul
    preinverted_jacobian <: Union{Val{false}, Val{true}}
    normal_form <: Union{Val{false}, Val{true}}
end

NonlinearSolveBase.@internal_caches DoglegCache :newton_cache :cauchy_cache

function InternalAPI.init(
        prob::AbstractNonlinearProblem, alg::Dogleg, J, fu, u;
        pre_inverted::Val = Val(false), linsolve_kwargs = (;),
        abstol = nothing, reltol = nothing, internalnorm::F = L2_NORM,
        shared::Val = Val(1), kwargs...
) where {F}
    newton_cache = InternalAPI.init(
        prob, alg.newton_descent, J, fu, u;
        pre_inverted, linsolve_kwargs, abstol, reltol, shared, kwargs...
    )
    cauchy_cache = InternalAPI.init(
        prob, alg.steepest_descent, J, fu, u;
        pre_inverted, linsolve_kwargs, abstol, reltol, shared, kwargs...
    )

    @bb δu = similar(u)
    δus = Utils.unwrap_val(shared) ≤ 1 ? nothing : map(2:Utils.unwrap_val(shared)) do i
        @bb δu_ = similar(u)
    end
    @bb δu_cache_1 = similar(u)
    @bb δu_cache_2 = similar(u)
    @bb δu_cache_mul = similar(u)

    normal_form = prob isa NonlinearLeastSquaresProblem &&
                  needs_square_A(alg.newton_descent.linsolve, u)

    Jᵀδu_cache = !normal_form ? J * Utils.safe_vec(δu) : nothing

    return DoglegCache(
        δu, δus, newton_cache, cauchy_cache, internalnorm, Jᵀδu_cache,
        δu_cache_1, δu_cache_2, δu_cache_mul, pre_inverted, Val(normal_form)
    )
end

# If trust_region is not specified, then use a Gauss-Newton step
function InternalAPI.solve!(
        cache::DoglegCache, J, fu, u, idx::Val = Val(1);
        trust_region = nothing, skip_solve::Bool = false, kwargs...
)
    @assert trust_region!==nothing "Trust Region must be specified for Dogleg. Use \
                                    `NewtonDescent` or `SteepestDescent` if you don't \
                                    want to use a Trust Region."

    δu = SciMLBase.get_du(cache, idx)
    T = promote_type(eltype(u), eltype(fu))
    δu_newton = InternalAPI.solve!(
        cache.newton_cache, J, fu, u, idx; skip_solve, kwargs...
    ).δu

    # Newton's Step within the trust region
    if cache.internalnorm(δu_newton) ≤ trust_region
        @bb copyto!(δu, δu_newton)
        set_du!(cache, δu, idx)
        return DescentResult(; δu, extras = (; δuJᵀJδu = T(NaN)))
    end

    # Take intersection of steepest descent direction and trust region if Cauchy point
    # lies outside of trust region
    if normal_form(cache)
        δu_cauchy = cache.newton_cache.Jᵀfu_cache
        JᵀJ = cache.newton_cache.JᵀJ_cache
        @bb @. δu_cauchy *= -1

        l_grad = cache.internalnorm(δu_cauchy)
        @bb cache.δu_cache_mul = JᵀJ × vec(δu_cauchy)
        δuJᵀJδu = Utils.dot(cache.δu_cache_mul, cache.δu_cache_mul)
    else
        δu_cauchy = InternalAPI.solve!(
            cache.cauchy_cache, J, fu, u, idx; skip_solve, kwargs...
        ).δu
        J_ = preinverted_jacobian(cache) ? inv(J) : J
        l_grad = cache.internalnorm(δu_cauchy)
        @bb cache.Jᵀδu_cache = J_ × vec(δu_cauchy)
        δuJᵀJδu = Utils.dot(cache.Jᵀδu_cache, cache.Jᵀδu_cache)
    end
    d_cauchy = (l_grad^3) / δuJᵀJδu

    if d_cauchy ≥ trust_region
        λ = trust_region / l_grad
        @bb @. δu = λ * δu_cauchy
        set_du!(cache, δu, idx)
        return DescentResult(; δu, extras = (; δuJᵀJδu = λ^2 * δuJᵀJδu))
    end

    # FIXME: For anything other than 2-norm a quadratic root will give incorrect results
    #        We need to do a local search with a interval root finding algorithm
    #        optimistix has a proper implementation for this
    # Take the intersection of dogleg with trust region if Cauchy point lies inside the
    # trust region
    @bb @. cache.δu_cache_1 = (d_cauchy / l_grad) * δu_cauchy
    @bb @. cache.δu_cache_2 = δu_newton - cache.δu_cache_1
    a = Utils.safe_dot(cache.δu_cache_2, cache.δu_cache_2)
    b = 2 * Utils.safe_dot(cache.δu_cache_1, cache.δu_cache_2)
    c = d_cauchy^2 - trust_region^2
    aux = max(0, b^2 - 4 * a * c)
    τ = (-b + sqrt(aux)) / (2 * a)

    @bb @. δu = cache.δu_cache_1 + τ * cache.δu_cache_2
    set_du!(cache, δu, idx)
    return DescentResult(; δu, extras = (; δuJᵀJδu = T(NaN)))
end
