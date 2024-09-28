"""
    Dogleg(; linsolve = nothing)

Switch between Newton's method and the steepest descent method depending on the size of the
trust region. The trust region is specified via keyword argument `trust_region` to
`solve!`.

See also [`SteepestDescent`](@ref), [`NewtonDescent`](@ref), [`DampedNewtonDescent`](@ref).
"""
@concrete struct Dogleg <: AbstractDescentAlgorithm
    newton_descent
    steepest_descent
end

function Base.show(io::IO, d::Dogleg)
    print(io,
        "Dogleg(newton_descent = $(d.newton_descent), steepest_descent = $(d.steepest_descent))")
end

supports_trust_region(::Dogleg) = true
get_linear_solver(alg::Dogleg) = get_linear_solver(alg.newton_descent)

function Dogleg(; linsolve = nothing, damping = False,
        damping_fn = missing, initial_damping = missing, kwargs...)
    if damping === False
        return Dogleg(NewtonDescent(; linsolve), SteepestDescent(; linsolve))
    end
    if damping_fn === missing || initial_damping === missing
        throw(ArgumentError("`damping_fn` and `initial_damping` must be supplied if \
                             `damping = Val(true)`."))
    end
    return Dogleg(DampedNewtonDescent(; linsolve, damping_fn, initial_damping),
        SteepestDescent(; linsolve))
end

@concrete mutable struct DoglegCache{pre_inverted, normalform} <: AbstractDescentCache
    δu
    δus
    newton_cache
    cauchy_cache
    internalnorm
    JᵀJ_cache
    δu_cache_1
    δu_cache_2
    δu_cache_mul
end

@internal_caches DoglegCache :newton_cache :cauchy_cache

function __internal_init(prob::AbstractNonlinearProblem, alg::Dogleg, J, fu, u;
        pre_inverted::Val{INV} = False, linsolve_kwargs = (;),
        abstol = nothing, reltol = nothing, internalnorm::F = DEFAULT_NORM,
        shared::Val{N} = Val(1), kwargs...) where {F, INV, N}
    newton_cache = __internal_init(prob, alg.newton_descent, J, fu, u; pre_inverted,
        linsolve_kwargs, abstol, reltol, shared, kwargs...)
    cauchy_cache = __internal_init(prob, alg.steepest_descent, J, fu, u; pre_inverted,
        linsolve_kwargs, abstol, reltol, shared, kwargs...)
    @bb δu = similar(u)
    δus = N ≤ 1 ? nothing : map(2:N) do i
        @bb δu_ = similar(u)
    end
    @bb δu_cache_1 = similar(u)
    @bb δu_cache_2 = similar(u)
    @bb δu_cache_mul = similar(u)

    T = promote_type(eltype(u), eltype(fu))

    normal_form = prob isa NonlinearLeastSquaresProblem &&
                  __needs_square_A(alg.newton_descent.linsolve, u)
    JᵀJ_cache = !normal_form ? J * _vec(δu) : nothing  # TODO: Rename

    return DoglegCache{INV, normal_form}(δu, δus, newton_cache, cauchy_cache, internalnorm,
        JᵀJ_cache, δu_cache_1, δu_cache_2, δu_cache_mul)
end

# If TrustRegion is not specified, then use a Gauss-Newton step
function __internal_solve!(cache::DoglegCache{INV, NF}, J, fu, u, idx::Val{N} = Val(1);
        trust_region = nothing, skip_solve::Bool = false, kwargs...) where {INV, NF, N}
    @assert trust_region!==nothing "Trust Region must be specified for Dogleg. Use \
                                    `NewtonDescent` or `SteepestDescent` if you don't \
                                    want to use a Trust Region."
    δu = get_du(cache, idx)
    T = promote_type(eltype(u), eltype(fu))
    δu_newton = __internal_solve!(
        cache.newton_cache, J, fu, u, idx; skip_solve, kwargs...).δu

    # Newton's Step within the trust region
    if cache.internalnorm(δu_newton) ≤ trust_region
        @bb copyto!(δu, δu_newton)
        set_du!(cache, δu, idx)
        return DescentResult(; δu, extras = (; δuJᵀJδu = T(NaN)))
    end

    # Take intersection of steepest descent direction and trust region if Cauchy point lies
    # outside of trust region
    if NF
        δu_cauchy = cache.newton_cache.Jᵀfu_cache
        JᵀJ = cache.newton_cache.JᵀJ_cache
        @bb @. δu_cauchy *= -1

        l_grad = cache.internalnorm(δu_cauchy)
        @bb cache.δu_cache_mul = JᵀJ × vec(δu_cauchy)
        δuJᵀJδu = __dot(δu_cauchy, cache.δu_cache_mul)
    else
        δu_cauchy = __internal_solve!(
            cache.cauchy_cache, J, fu, u, idx; skip_solve, kwargs...).δu
        J_ = INV ? inv(J) : J
        l_grad = cache.internalnorm(δu_cauchy)
        @bb cache.JᵀJ_cache = J × vec(δu_cauchy)  # TODO: Rename
        δuJᵀJδu = __dot(cache.JᵀJ_cache, cache.JᵀJ_cache)
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
    a = dot(cache.δu_cache_2, cache.δu_cache_2)
    b = 2 * dot(cache.δu_cache_1, cache.δu_cache_2)
    c = d_cauchy^2 - trust_region^2
    aux = max(0, b^2 - 4 * a * c)
    τ = (-b + sqrt(aux)) / (2 * a)

    @bb @. δu = cache.δu_cache_1 + τ * cache.δu_cache_2
    set_du!(cache, δu, idx)
    return DescentResult(; δu, extras = (; δuJᵀJδu = T(NaN)))
end
