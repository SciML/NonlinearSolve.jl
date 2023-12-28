"""
    Dogleg(; linsolve = nothing, precs = DEFAULT_PRECS)

Switch between Newton's method and the steepest descent method depending on the size of the
trust region. The trust region is specified via keyword argument `trust_region` to
`solve!`.

### Keyword Arguments

  - `linsolve`: the [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl) used for the
    linear solves within the Newton method. Defaults to `nothing`, which means it uses the
    LinearSolve.jl default algorithm choice. For more information on available algorithm
    choices, see the
    [LinearSolve.jl documentation](https://docs.sciml.ai/LinearSolve/stable/).
  - `precs`: the choice of preconditioners for the linear solver. Defaults to using no
    preconditioners. For more information on specifying preconditioners for LinearSolve
    algorithms, consult the
    [LinearSolve.jl documentation](https://docs.sciml.ai/LinearSolve/stable/).

See also [`SteepestDescent`](@ref), [`NewtonDescent`](@ref), [`DampedNewtonDescent`](@ref).
"""
@concrete struct Dogleg <: AbstractDescentAlgorithm
    newton_descent
    steepest_descent
end

supports_trust_region(::Dogleg) = true

function Dogleg(; linsolve = nothing, precs = DEFAULT_PRECS, kwargs...)
    return Dogleg(NewtonDescent(; linsolve, precs), SteepestDescent())
end

@concrete mutable struct DoglegCache{pre_inverted, normalform,
    NC <: NewtonDescentCache{pre_inverted, normalform},
    CC <: SteepestDescentCache{pre_inverted}} <: AbstractDescentCache
    δu
    newton_cache::NC
    cauchy_cache::CC
    internalnorm
    JᵀJ_cache
    δu_cache_1
    δu_cache_2
    δu_cache_mul
    prev_d_cauchy
    prev_l_grad
    prev_a
    prev_b
end

function init_cache(prob::NonlinearProblem, alg::Dogleg, J, fu, u;
        pre_inverted::Val{INV} = False, linsolve_kwargs = (;), abstol = nothing,
        reltol = nothing, internalnorm::F = DEFAULT_NORM, kwargs...) where {F, INV}
    @warn "Setting `pre_inverted = Val(true)` for `Dogleg` is not recommended." maxlog=1
    newton_cache = init_cache(prob, alg.newton_descent, J, fu, u; pre_inverted,
        linsolve_kwargs, abstol, reltol, kwargs...)
    cauchy_cache = init_cache(prob, alg.steepest_descent, J, fu, u; pre_inverted,
        linsolve_kwargs, abstol, reltol, kwargs...)
    @bb δu = similar(u)
    @bb δu_cache_1 = similar(u)
    @bb δu_cache_2 = similar(u)
    @bb δu_cache_mul = similar(u)

    T = promote_type(eltype(u), eltype(fu))
    
    normal_form = __needs_square_A(alg.linsolve, u)
    JᵀJ_cache = !normal_form ? transpose(J) * J : nothing

    return DoglegCache{INV, normal_form}(δu, newton_cache, cauchy_cache, internalnorm,
        JᵀJ_cache,  δu_cache_1, δu_cache_2, δu_cache_mul, T(0), T(0), T(0), T(0))
end

# If TrustRegion is not specified, then use a Gauss-Newton step
function SciMLBase.solve!(cache::DoglegCache{INV, NF}, J, fu; trust_region = nothing,
        skip_solve::Bool = false, kwargs...) where {INV, NF}
    @assert trust_region === nothing "Trust Region must be specified for Dogleg. Use \
                                      `NewtonDescent` or `SteepestDescent` if you don't \
                                      want to use a Trust Region."
    δu_newton = solve!(cache.newton_cache, J, fu; skip_solve, kwargs...)

    # Newton's Step within the trust region
    if cache.internalnorm(δu_newton) ≤ trust_region
        @bb copyto!(cache.δu, δu_newton)
        return cache.δu
    end

    # Take intersection of steepest descent direction and trust region if Cauchy point lies
    # outside of trust region
    if NF
        δu_cauchy = cache.newton_cache.Jᵀfu_cache
        JᵀJ = cache.newton_cache.JᵀJ_cache
        @bb @. δu_cauchy *= -1
    else
        δu_cauchy = solve!(cache.cauchy_cache, J, fu; skip_solve, kwargs...)
        if !skip_solve
            J_ = INV ? inv(J) : J
            @bb cache.JᵀJ_cache = transpose(J_) × J_
        end
        JᵀJ = cache.JᵀJ_cache
    end

    if skip_solve
        d_cauchy = cache.prev_d_cauchy
        l_grad = cache.prev_l_grad
    else
        l_grad = cache.internalnorm(δu_cauchy)
        @bb cache.δu_cache_mul = JᵀJ × vec(δu_cauchy)
        d_cauchy = (l_grad^3) / dot(_vec(δu_cauchy), cache.δu_cache_mul)
    end

    if d_cauchy ≥ trust_region
        @bb @. cache.δu = (trust_region / l_grad) * δu_cauchy
        return cache.δu
    end

    # FIXME: For anything other than 2-norm a quadratic root will give incorrect results
    #        We need to do a local search with a interval root finding algorithm
    #        optimistix has a proper implementation for this
    # Take the intersection of dogleg with trust region if Cauchy point lies inside the
    # trust region
    if !skip_solve
        @bb @. cache.δu_cache_1 = (d_cauchy / l_grad) * δu_cauchy
        @bb @. cache.δu_cache_2 = δu_newton - cache.δu_cache_1
        a = dot(_vec(cache.δu_cache_2), _vec(cache.δu_cache_2))
        b = 2 * dot(_vec(cache.δu_cache_1), _vec(cache.δu_cache_2))
    else
        a = cache.prev_a
        b = cache.prev_b
    end
    c = d_cauchy^2 - trust_region^2
    aux = max(0, b^2 - 4 * a * c)
    τ = (-b + sqrt(aux)) / (2 * a)

    @bb @. cache.δu = cache.δu_cache_1 + τ * cache.δu_cache_2
    return cache.δu
end
