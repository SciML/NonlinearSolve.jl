"""
    Dogleg(; linsolve = nothing)

Switch between Newton's method and the steepest descent method depending on the size of the
trust region. The trust region is specified via keyword argument `trust_region` to
`solve!`. A `trust_region_scaling` vector `d` measures the region as `‖d .* δu‖ ≤ Δ`
and makes the steepest-descent leg the gradient in that norm, `-(Jᵀfu) ./ d.^2`, as in
MINPACK's `dogleg` and NLsolve's `trust_region` with `autoscale`.

See also [`SteepestDescent`](@ref), [`NewtonDescent`](@ref), [`DampedNewtonDescent`](@ref).
"""
@concrete struct Dogleg <: AbstractDescentDirection
    newton_descent <: Union{NewtonDescent, DampedNewtonDescent}
    steepest_descent <: SteepestDescent
end

supports_trust_region(::Dogleg) = true
supports_trust_region_scaling(::Dogleg) = true
get_linear_solver(alg::Dogleg) = get_linear_solver(alg.newton_descent)

function Dogleg(;
        linsolve = nothing, damping = Val(false),
        damping_fn = missing, initial_damping = missing, kwargs...
    )
    if !Utils.unwrap_val(damping)
        return Dogleg(NewtonDescent(; linsolve), SteepestDescent(; linsolve))
    end
    if damping_fn === missing || initial_damping === missing
        throw(ArgumentError("`damping_fn` and `initial_damping` must be supplied if \
                             `damping = Val(true)`."))
    end
    return Dogleg(
        DampedNewtonDescent(; linsolve, damping_fn, initial_damping),
        SteepestDescent(; linsolve)
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
    scaled_cache
    preinverted_jacobian <: Union{Val{false}, Val{true}}
    normal_form <: Union{Val{false}, Val{true}}
end

@internal_caches DoglegCache :newton_cache :cauchy_cache

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

    @bb δu = zero(u)
    δus = Utils.unwrap_val(shared) ≤ 1 ? nothing : map(2:Utils.unwrap_val(shared)) do i
            @bb δu_ = zero(u)
    end
    @bb δu_cache_1 = similar(u)
    @bb δu_cache_2 = similar(u)
    @bb δu_cache_mul = similar(u)
    @bb scaled_cache = similar(u)

    normal_form = prob isa NonlinearLeastSquaresProblem &&
        needs_square_A(alg.newton_descent.linsolve, u)

    Jᵀδu_cache = !normal_form ? J * Utils.safe_vec(δu) : nothing

    return DoglegCache(
        δu, δus, newton_cache, cauchy_cache, internalnorm, Jᵀδu_cache,
        δu_cache_1, δu_cache_2, δu_cache_mul, scaled_cache, pre_inverted, Val(normal_form)
    )
end

# `‖d .* x‖`, the trust region's norm; `d === nothing` is the unscaled norm
scaled_norm(cache::DoglegCache, ::Nothing, x) = cache.internalnorm(x)
function scaled_norm(cache::DoglegCache, d, x)
    @bb @. cache.scaled_cache = d * x
    return cache.internalnorm(cache.scaled_cache)
end

# The steepest-descent direction in the scaled norm, `-(Jᵀfu) ./ d.^2`, from `-(Jᵀfu)` in place
scale_gradient!!(::Nothing, g) = g
function scale_gradient!!(d, g)
    @bb @. g = g / d^2
    return g
end

# `(⟨d .* x, d .* y⟩, ‖d .* y‖²)` in the Euclidean inner product
scaled_dots(cache::DoglegCache, ::Nothing, x, y) = (Utils.safe_dot(x, y), Utils.safe_dot(y, y))
function scaled_dots(cache::DoglegCache, d, x, y)
    @bb @. cache.scaled_cache = d^2 * y
    return Utils.safe_dot(x, cache.scaled_cache), Utils.safe_dot(y, cache.scaled_cache)
end

# If trust_region is not specified, then use a Gauss-Newton step
function InternalAPI.solve!(
        cache::DoglegCache, J, fu, u, idx::Val = Val(1);
        trust_region = nothing, trust_region_scaling = nothing, skip_solve::Bool = false,
        kwargs...
    )
    @assert trust_region !== nothing "Trust Region must be specified for Dogleg. Use \
                                    `NewtonDescent` or `SteepestDescent` if you don't \
                                    want to use a Trust Region."

    δu = SciMLBase.get_du(cache, idx)
    T = promote_type(eltype(u), eltype(fu))
    δu_newton = InternalAPI.solve!(
        cache.newton_cache, J, fu, u, idx; skip_solve, kwargs...
    ).δu

    # Newton's Step within the trust region
    d = trust_region_scaling
    if scaled_norm(cache, d, δu_newton) ≤ trust_region
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
        δu_cauchy = scale_gradient!!(d, δu_cauchy)

        l_grad = scaled_norm(cache, d, δu_cauchy)
        @bb cache.δu_cache_mul = JᵀJ × vec(δu_cauchy)
        δuJᵀJδu = Utils.safe_dot(δu_cauchy, cache.δu_cache_mul)
    else
        δu_cauchy = InternalAPI.solve!(
            cache.cauchy_cache, J, fu, u, idx; skip_solve, kwargs...
        ).δu
        δu_cauchy = scale_gradient!!(d, δu_cauchy)
        J_ = preinverted_jacobian(cache) ? inv(J) : J
        l_grad = scaled_norm(cache, d, δu_cauchy)
        @bb cache.Jᵀδu_cache = J_ × vec(δu_cauchy)
        δuJᵀJδu = Utils.safe_dot(cache.Jᵀδu_cache, cache.Jᵀδu_cache)
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
    c1ᵀc2, a = scaled_dots(cache, d, cache.δu_cache_1, cache.δu_cache_2)
    b = 2 * c1ᵀc2
    c = d_cauchy^2 - trust_region^2
    aux = max(0, b^2 - 4 * a * c)
    τ = (-b + sqrt(aux)) / (2 * a)

    @bb @. δu = cache.δu_cache_1 + τ * cache.δu_cache_2
    set_du!(cache, δu, idx)
    return DescentResult(; δu, extras = (; δuJᵀJδu = T(NaN)))
end
