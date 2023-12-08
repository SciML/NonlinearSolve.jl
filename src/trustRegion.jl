"""
    RadiusUpdateSchemes

`RadiusUpdateSchemes` is the standard enum interface for different types of radius update
schemes implemented in the Trust Region method. These schemes specify how the radius of the
so-called trust region is updated after each iteration of the algorithm. The specific role
and caveats associated with each scheme are provided below.

## Using `RadiusUpdateSchemes`

`RadiusUpdateSchemes` uses the standard
[EnumX Interface](https://github.com/fredrikekre/EnumX.jl), and hence inherits all
properties of being an EnumX, including the type of each constituent enum states as
`RadiusUpdateSchemes.T`. Simply put the desired scheme as follows:
`TrustRegion(radius_update_scheme = your desired update scheme)`. For example,
`sol = solve(prob, alg=TrustRegion(radius_update_scheme = RadiusUpdateSchemes.Hei))`.
"""
@enumx RadiusUpdateSchemes begin
    """
        RadiusUpdateSchemes.Simple

    The simple or conventional radius update scheme. This scheme is chosen by default and
    follows the conventional approach to update the trust region radius, i.e. if the trial
    step is accepted it increases the radius by a fixed factor (bounded by a maximum radius)
    and if the trial step is rejected, it shrinks the radius by a fixed factor.
    """
    Simple

    """
        RadiusUpdateSchemes.NLsolve

    The same updating scheme as in NLsolve's (https://github.com/JuliaNLSolvers/NLsolve.jl)
    trust region dogleg implementation.
    """
    NLsolve

    """
        RadiusUpdateSchemes.NocedalWright

    Trust region updating scheme as in Nocedal and Wright [see Alg 11.5, page 291].
    """
    NocedalWright

    """
        RadiusUpdateSchemes.Hei

    This scheme is proposed by Hei, L. [1]. The trust region radius depends on the size
    (norm) of the current step size. The hypothesis is to let the radius converge to zero as
    the iterations progress, which is more reliable and robust for ill-conditioned as well
    as degenerate problems.

    [1] Hei, Long. "A self-adaptive trust region algorithm." Journal of Computational
    Mathematics (2003): 229-236.
    """
    Hei

    """
        RadiusUpdateSchemes.Yuan

    This scheme is proposed by Yuan, Y [1]. Similar to Hei's scheme, the trust region is
    updated in a way so that it converges to zero, however here, the radius depends on the
    size (norm) of the current gradient of the objective (merit) function. The hypothesis is
    that the step size is bounded by the gradient size, so it makes sense to let the radius
    depend on the gradient.

    [1] Fan, Jinyan, Jianyu Pan, and Hongyan Song. "A retrospective trust region algorithm
    with trust region converging to zero." Journal of Computational Mathematics 34.4 (2016):
    421-436.
    """
    Yuan

    """
        RadiusUpdateSchemes.Bastin

    This scheme is proposed by Bastin, et al. [1]. The scheme is called a retrospective
    update scheme as it uses the model function at the current iteration to compute the
    ratio of the actual reduction and the predicted reduction in the previous trial step,
    and use this ratio to update the trust region radius. The hypothesis is to exploit the
    information made available during the optimization process in order to vary the accuracy
    of the objective function computation.

    [1] Bastin, Fabian, et al. "A retrospective trust-region method for unconstrained
    optimization." Mathematical programming 123 (2010): 395-418.
    """
    Bastin

    """
        RadiusUpdateSchemes.Fan

    This scheme is proposed by Fan, J. [1]. It is very much similar to Hei's and Yuan's
    schemes as it lets the trust region radius depend on the current size (norm) of the
    objective (merit) function itself. These new update schemes are known to improve local
    convergence.

    [1] Fan, Jinyan. "Convergence rate of the trust region method for nonlinear equations
    under local error bound condition." Computational Optimization and Applications 34.2
    (2006): 215-227.
    """
    Fan
end

"""
    TrustRegion(; concrete_jac = nothing, linsolve = nothing, precs = DEFAULT_PRECS,
        radius_update_scheme::RadiusUpdateSchemes.T = RadiusUpdateSchemes.Simple,
        max_trust_radius::Real = 0 // 1, initial_trust_radius::Real = 0 // 1,
        step_threshold::Real = 1 // 10, shrink_threshold::Real = 1 // 4,
        expand_threshold::Real = 3 // 4, shrink_factor::Real = 1 // 4,
        expand_factor::Real = 2 // 1, max_shrink_times::Int = 32, adkwargs...)

An advanced TrustRegion implementation with support for efficient handling of sparse
matrices via colored automatic differentiation and preconditioned linear solvers. Designed
for large-scale and numerically-difficult nonlinear systems.

### Keyword Arguments

  - `autodiff`: determines the backend used for the Jacobian. Note that this argument is
    ignored if an analytical Jacobian is passed, as that will be used instead. Defaults to
    `nothing` which means that a default is selected according to the problem specification!.
    Valid choices are types from ADTypes.jl.
  - `concrete_jac`: whether to build a concrete Jacobian. If a Krylov-subspace method is used,
    then the Jacobian will not be constructed and instead direct Jacobian-vector products
    `J*v` are computed using forward-mode automatic differentiation or finite differencing
    tricks (without ever constructing the Jacobian). However, if the Jacobian is still needed,
    for example for a preconditioner, `concrete_jac = true` can be passed in order to force
    the construction of the Jacobian.
  - `linsolve`: the [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl) used for the
    linear solves within the Newton method. Defaults to `nothing`, which means it uses the
    LinearSolve.jl default algorithm choice. For more information on available algorithm
    choices, see the [LinearSolve.jl documentation](https://docs.sciml.ai/LinearSolve/stable/).
  - `precs`: the choice of preconditioners for the linear solver. Defaults to using no
    preconditioners. For more information on specifying preconditioners for LinearSolve
    algorithms, consult the
    [LinearSolve.jl documentation](https://docs.sciml.ai/LinearSolve/stable/).
  - `radius_update_scheme`: the choice of radius update scheme to be used. Defaults to `RadiusUpdateSchemes.Simple`
    which follows the conventional approach. Other available schemes are `RadiusUpdateSchemes.Hei`,
    `RadiusUpdateSchemes.Yuan`, `RadiusUpdateSchemes.Bastin`, `RadiusUpdateSchemes.Fan`. These schemes
    have the trust region radius converging to zero that is seen to improve convergence. For more details, see the
    [Yuan, Yx](https://link.springer.com/article/10.1007/s10107-015-0893-2#Sec4).
  - `max_trust_radius`: the maximal trust region radius.
    Defaults to `max(norm(fu), maximum(u) - minimum(u))`.
  - `initial_trust_radius`: the initial trust region radius. Defaults to
    `max_trust_radius / 11`.
  - `step_threshold`: the threshold for taking a step. In every iteration, the threshold is
    compared with a value `r`, which is the actual reduction in the objective function divided
    by the predicted reduction. If `step_threshold > r` the model is not a good approximation,
    and the step is rejected. Defaults to `0.1`. For more details, see
    [Rahpeymaii, F.](https://link.springer.com/article/10.1007/s40096-020-00339-4)
  - `shrink_threshold`: the threshold for shrinking the trust region radius. In every
    iteration, the threshold is compared with a value `r` which is the actual reduction in the
    objective function divided by the predicted reduction. If `shrink_threshold > r` the trust
    region radius is shrunk by `shrink_factor`. Defaults to `0.25`. For more details, see
    [Rahpeymaii, F.](https://link.springer.com/article/10.1007/s40096-020-00339-4)
  - `expand_threshold`: the threshold for expanding the trust region radius. If a step is
    taken, i.e `step_threshold < r` (with `r` defined in `shrink_threshold`), a check is also
    made to see if `expand_threshold < r`. If that is true, the trust region radius is
    expanded by `expand_factor`. Defaults to `0.75`.
  - `shrink_factor`: the factor to shrink the trust region radius with if
    `shrink_threshold > r` (with `r` defined in `shrink_threshold`). Defaults to `0.25`.
  - `expand_factor`: the factor to expand the trust region radius with if
    `expand_threshold < r` (with `r` defined in `shrink_threshold`). Defaults to `2.0`.
  - `max_shrink_times`: the maximum number of times to shrink the trust region radius in a
    row, `max_shrink_times` is exceeded, the algorithm returns. Defaults to `32`.
  - `vjp_autodiff`: Automatic Differentiation Backend used for vector-jacobian products.
    This is applicable if the linear solver doesn't require a concrete jacobian, for eg.,
    Krylov Methods. Defaults to `nothing`, which means if the problem is out of place and
    `Zygote` is loaded then, we use `AutoZygote`. In all other, cases `FiniteDiff` is used.
"""
@concrete struct TrustRegion{CJ, AD, MTR} <: AbstractNewtonAlgorithm{CJ, AD}
    ad::AD
    linsolve
    precs
    radius_update_scheme::RadiusUpdateSchemes.T
    max_trust_radius
    initial_trust_radius::MTR
    step_threshold::MTR
    shrink_threshold::MTR
    expand_threshold::MTR
    shrink_factor::MTR
    expand_factor::MTR
    max_shrink_times::Int
    vjp_autodiff
end

function set_ad(alg::TrustRegion{CJ}, ad) where {CJ}
    return TrustRegion{CJ}(ad, alg.linsolve, alg.precs, alg.radius_update_scheme,
        alg.max_trust_radius, alg.initial_trust_radius, alg.step_threshold,
        alg.shrink_threshold, alg.expand_threshold, alg.shrink_factor, alg.expand_factor,
        alg.max_shrink_times, alg.vjp_autodiff)
end

function TrustRegion(; concrete_jac = nothing, linsolve = nothing, precs = DEFAULT_PRECS,
        radius_update_scheme::RadiusUpdateSchemes.T = RadiusUpdateSchemes.Simple,
        max_trust_radius::Real = 0 // 1, initial_trust_radius::Real = 0 // 1,
        step_threshold::Real = 1 // 10000, shrink_threshold::Real = 1 // 4,
        expand_threshold::Real = 3 // 4, shrink_factor::Real = 1 // 4,
        expand_factor::Real = 2 // 1, max_shrink_times::Int = 32, vjp_autodiff = nothing,
        autodiff = nothing)
    return TrustRegion{_unwrap_val(concrete_jac)}(autodiff, linsolve, precs,
        radius_update_scheme, max_trust_radius, initial_trust_radius, step_threshold,
        shrink_threshold, expand_threshold, shrink_factor, expand_factor, max_shrink_times,
        vjp_autodiff)
end

@concrete mutable struct TrustRegionCache{iip} <: AbstractNonlinearSolveCache{iip}
    f
    alg
    u
    u_cache
    u_cache_2
    u_gauss_newton
    u_cauchy
    fu
    fu_cache
    fu_cache_2
    J
    J_cache
    JᵀJ
    Jᵀf
    p
    uf
    du
    lr_mul_cache
    linsolve
    jac_cache
    force_stop::Bool
    maxiters::Int
    internalnorm
    retcode::ReturnCode.T
    abstol
    reltol
    prob
    radius_update_scheme::RadiusUpdateSchemes.T
    trust_r
    max_trust_r
    step_threshold
    shrink_threshold
    expand_threshold
    shrink_factor
    expand_factor
    loss
    loss_new
    shrink_counter::Int
    make_new_J::Bool
    r
    p1
    p2
    p3
    p4
    ϵ
    jvp_operator  # For Yuan
    stats::NLStats
    tc_cache
    trace
end

function SciMLBase.__init(prob::NonlinearProblem{uType, iip}, alg_::TrustRegion, args...;
        alias_u0 = false, maxiters = 1000, abstol = nothing, reltol = nothing,
        termination_condition = nothing, internalnorm = DEFAULT_NORM,
        linsolve_kwargs = (;), kwargs...) where {uType, iip}
    alg = get_concrete_algorithm(alg_, prob)
    @unpack f, u0, p = prob
    u = __maybe_unaliased(u0, alias_u0)
    @bb u_cache = copy(u)
    @bb u_cache_2 = similar(u)
    fu = evaluate_f(prob, u)
    @bb fu_cache_2 = zero(fu)

    loss = __trust_region_loss(internalnorm, fu)

    uf, _, J, fu_cache, jac_cache, du, JᵀJ, Jᵀf = jacobian_caches(alg, f, u, p, Val(iip);
        linsolve_kwargs, linsolve_with_JᵀJ = Val(true), lininit = Val(false))
    linsolve = linsolve_caches(J, fu_cache, du, p, alg)

    @bb u_cache_2 = similar(u)
    @bb u_cauchy = similar(u)
    @bb u_gauss_newton = similar(u)
    J_cache = J isa SciMLOperators.AbstractSciMLOperator ||
              setindex_trait(J) === CannotSetindex() ? J : similar(J)
    @bb lr_mul_cache = similar(du)

    loss_new = loss
    shrink_counter = 0
    make_new_J = true
    r = loss

    floatType = typeof(r)

    # set trust region update scheme
    radius_update_scheme = alg.radius_update_scheme

    # set default type for all trust region parameters
    trustType = floatType
    if radius_update_scheme == RadiusUpdateSchemes.NLsolve
        max_trust_radius = convert(trustType, Inf)
        initial_trust_radius = internalnorm(u0) > 0 ? convert(trustType, internalnorm(u0)) :
                               one(trustType)
    else
        max_trust_radius = convert(trustType, alg.max_trust_radius)
        if iszero(max_trust_radius)
            max_trust_radius = convert(trustType,
                max(internalnorm(fu), maximum(u) - minimum(u)))
        end
        initial_trust_radius = convert(trustType, alg.initial_trust_radius)
        if iszero(initial_trust_radius)
            initial_trust_radius = convert(trustType, max_trust_radius / 11)
        end
    end
    step_threshold = convert(trustType, alg.step_threshold)
    shrink_threshold = convert(trustType, alg.shrink_threshold)
    expand_threshold = convert(trustType, alg.expand_threshold)
    shrink_factor = convert(trustType, alg.shrink_factor)
    expand_factor = convert(trustType, alg.expand_factor)

    # Parameters for the Schemes
    p1 = convert(floatType, 0.0)
    p2 = convert(floatType, 0.0)
    p3 = convert(floatType, 0.0)
    p4 = convert(floatType, 0.0)
    ϵ = convert(floatType, 1.0e-8)
    jvp_operator = nothing
    if radius_update_scheme === RadiusUpdateSchemes.NLsolve
        p1 = convert(floatType, 0.5)
    elseif radius_update_scheme === RadiusUpdateSchemes.Hei
        step_threshold = convert(trustType, 0.0)
        shrink_threshold = convert(trustType, 0.25)
        expand_threshold = convert(trustType, 0.25)
        p1 = convert(floatType, 5.0)  # M
        p2 = convert(floatType, 0.1)  # β
        p3 = convert(floatType, 0.15) # γ1
        p4 = convert(floatType, 0.15) # γ2
        initial_trust_radius = convert(trustType, 1.0)
    elseif radius_update_scheme === RadiusUpdateSchemes.Yuan
        step_threshold = convert(trustType, 0.0001)
        shrink_threshold = convert(trustType, 0.25)
        expand_threshold = convert(trustType, 0.25)
        p1 = convert(floatType, 2.0)   # μ
        p2 = convert(floatType, 1 / 6) # c5
        p3 = convert(floatType, 6.0)   # c6
        jvp_operator = __jacvec(uf, u; fu, autodiff = __get_nonsparse_ad(alg.ad))
        @bb Jᵀf = jvp_operator × fu
        initial_trust_radius = convert(trustType, p1 * internalnorm(Jᵀf))
    elseif radius_update_scheme === RadiusUpdateSchemes.Fan
        step_threshold = convert(trustType, 0.0001)
        shrink_threshold = convert(trustType, 0.25)
        expand_threshold = convert(trustType, 0.75)
        p1 = convert(floatType, 0.1) # μ
        p2 = convert(floatType, 0.25) # c5
        p3 = convert(floatType, 12.0) # c6
        p4 = convert(floatType, 1.0e18) # M
        initial_trust_radius = convert(trustType, p1 * (internalnorm(fu)^0.99))
    elseif radius_update_scheme === RadiusUpdateSchemes.Bastin
        step_threshold = convert(trustType, 0.05)
        shrink_threshold = convert(trustType, 0.05)
        expand_threshold = convert(trustType, 0.9)
        p1 = convert(floatType, 2.5)  # alpha_1
        p2 = convert(floatType, 0.25) # alpha_2
        initial_trust_radius = convert(trustType, 1.0)
    end

    abstol, reltol, tc_cache = init_termination_cache(abstol, reltol, fu, u,
        termination_condition)
    trace = init_nonlinearsolve_trace(alg, u, fu, ApplyArray(__zero, J), du; kwargs...)

    return TrustRegionCache{iip}(f, alg, u, u_cache, u_cache_2, u_gauss_newton, u_cauchy,
        fu, fu_cache, fu_cache_2, J, J_cache, JᵀJ, Jᵀf, p, uf, du, lr_mul_cache, linsolve,
        jac_cache, false, maxiters, internalnorm, ReturnCode.Default, abstol, reltol, prob,
        radius_update_scheme, initial_trust_radius, max_trust_radius, step_threshold,
        shrink_threshold, expand_threshold, shrink_factor, expand_factor, loss, loss_new,
        shrink_counter, make_new_J, r, p1, p2, p3, p4, ϵ, jvp_operator,
        NLStats(1, 0, 0, 0, 0), tc_cache, trace)
end

function perform_step!(cache::TrustRegionCache{iip}) where {iip}
    if cache.make_new_J
        cache.J = jacobian!!(cache.J, cache)

        __update_JᵀJ!(cache)
        __update_Jᵀf!(cache)

        # do not use A = cache.H, b = _vec(cache.g) since it is equivalent
        # to  A = cache.J, b = _vec(fu) as long as the Jacobian is non-singular
        linres = dolinsolve(cache, cache.alg.precs, cache.linsolve, A = cache.J,
            b = _vec(cache.fu), linu = _vec(cache.u_gauss_newton), p = cache.p,
            reltol = cache.abstol)
        cache.linsolve = linres.cache
        cache.u_gauss_newton = _restructure(cache.u_gauss_newton, linres.u)
        @bb @. cache.u_gauss_newton *= -1
    end

    # compute dogleg step
    dogleg!(cache)

    # compute the potentially new u
    @bb @. cache.u_cache_2 = cache.u + cache.du
    evaluate_f(cache, cache.u_cache_2, cache.p, Val{:fu_cache_2}())
    trust_region_step!(cache)
    return nothing
end

function retrospective_step!(cache::TrustRegionCache{iip}) where {iip}
    J = jacobian!!(cache.J_cache, cache)
    __update_JᵀJ!(cache, J)
    __update_Jᵀf!(cache, J)

    num = __trust_region_loss(cache, cache.fu) - __trust_region_loss(cache, cache.fu_cache)
    denom = dot(_vec(cache.du), _vec(cache.Jᵀf)) + __lr_mul(cache, cache.JᵀJ, cache.du) / 2
    return num / denom
end

function trust_region_step!(cache::TrustRegionCache)
    cache.loss_new = __trust_region_loss(cache, cache.fu_cache_2)

    # Compute the ratio of the actual reduction to the predicted reduction.
    cache.r = -(cache.loss - cache.loss_new) /
              (dot(_vec(cache.du), _vec(cache.Jᵀf)) +
               __lr_mul(cache, cache.JᵀJ, _vec(cache.du)) / 2)

    @unpack r, radius_update_scheme = cache
    make_new_J = false
    if r > cache.step_threshold
        take_step!(cache)
        cache.loss = cache.loss_new
        make_new_J = true
    end

    if radius_update_scheme === RadiusUpdateSchemes.Simple
        if r < cache.shrink_threshold
            cache.trust_r *= cache.shrink_factor
            cache.shrink_counter += 1
        else
            cache.shrink_counter = 0
            if r > cache.step_threshold && r > cache.expand_threshold
                cache.trust_r = min(cache.expand_factor * cache.trust_r, cache.max_trust_r)
            end
        end
    elseif radius_update_scheme === RadiusUpdateSchemes.NLsolve
        if r < 1 // 10
            cache.shrink_counter += 1
            cache.trust_r *= 1 // 2
        else
            cache.shrink_counter = 0
            if r ≥ 9 // 10
                cache.trust_r = 2 * cache.internalnorm(cache.du)
            elseif r ≥ 1 // 2
                cache.trust_r = max(cache.trust_r, 2 * cache.internalnorm(cache.du))
            end
        end
    elseif radius_update_scheme === RadiusUpdateSchemes.NocedalWright
        if r < 1 // 4
            cache.shrink_counter += 1
            cache.trust_r = (1 // 4) * cache.internalnorm(cache.du)
        else
            cache.shrink_counter = 0
            if r > 3 // 4 &&
               abs(cache.internalnorm(cache.du) - cache.trust_r) < 1e-6 * cache.trust_r
                cache.trust_r = min(2 * cache.trust_r, cache.max_trust_r)
            end
        end
    elseif radius_update_scheme === RadiusUpdateSchemes.Hei
        @unpack shrink_threshold, p1, p2, p3, p4 = cache
        tr_new = __rfunc(r, shrink_threshold, p1, p3, p4, p2) * cache.internalnorm(cache.du)
        if tr_new < cache.trust_r
            cache.shrink_counter += 1
        else
            cache.shrink_counter = 0
        end
        cache.trust_r = tr_new

        cache.internalnorm(cache.Jᵀf) < cache.ϵ && (cache.force_stop = true)
    elseif radius_update_scheme === RadiusUpdateSchemes.Yuan
        if r < cache.shrink_threshold
            cache.p1 = cache.p2 * cache.p1
            cache.shrink_counter += 1
        else
            if r ≥ cache.expand_threshold &&
               cache.internalnorm(cache.du) > cache.trust_r / 2
                cache.p1 = cache.p3 * cache.p1
            end
            cache.shrink_counter = 0
        end

        @bb cache.Jᵀf = cache.jvp_operator × vec(cache.fu)
        cache.trust_r = cache.p1 * cache.internalnorm(cache.Jᵀf)

        cache.internalnorm(cache.Jᵀf) < cache.ϵ && (cache.force_stop = true)
    elseif radius_update_scheme === RadiusUpdateSchemes.Fan
        if r < cache.shrink_threshold
            cache.p1 *= cache.p2
            cache.shrink_counter += 1
        else
            cache.shrink_counter = 0
            r > cache.expand_threshold && (cache.p1 = min(cache.p1 * cache.p3, cache.p4))
        end
        cache.trust_r = cache.p1 * (cache.internalnorm(cache.fu)^0.99)
        cache.internalnorm(cache.Jᵀf) < cache.ϵ && (cache.force_stop = true)
    elseif radius_update_scheme === RadiusUpdateSchemes.Bastin
        if r > cache.step_threshold
            if retrospective_step!(cache) ≥ cache.expand_threshold
                cache.trust_r = max(cache.p1 * cache.internalnorm(cache.du), cache.trust_r)
            end
            cache.shrink_counter = 0
        else
            cache.trust_r *= cache.p2
            cache.shrink_counter += 1
        end
    end

    update_trace!(cache.trace, cache.stats.nsteps + 1, cache.u, cache.fu, cache.J,
        @~(cache.u.-cache.u_cache))
    check_and_update!(cache, cache.fu, cache.u, cache.u_cache)
end

function dogleg!(cache::TrustRegionCache{iip}) where {iip}
    # Take the full Gauss-Newton step if lies within the trust region.
    if cache.internalnorm(cache.u_gauss_newton) ≤ cache.trust_r
        @bb copyto!(cache.du, cache.u_gauss_newton)
        return
    end

    # Take intersection of steepest descent direction and trust region if Cauchy point lies
    # outside of trust region
    l_grad = cache.internalnorm(cache.Jᵀf) # length of the gradient
    d_cauchy = l_grad^3 / __lr_mul(cache)
    g = _restructure(cache.du, cache.Jᵀf)
    if d_cauchy ≥ cache.trust_r
        # step to the end of the trust region
        @bb @. cache.du = -(cache.trust_r / l_grad) * g
        return
    end

    # Take the intersection of dogleg with trust region if Cauchy point lies inside the
    # trust region
    @bb @. cache.u_cauchy = -(d_cauchy / l_grad) * g # compute Cauchy point
    @bb @. cache.u_cache_2 = cache.u_gauss_newton - cache.u_cauchy # calf of the dogleg

    a = dot(cache.u_cache_2, cache.u_cache_2)
    b = 2 * dot(cache.u_cauchy, cache.u_cache_2)
    c = d_cauchy^2 - cache.trust_r^2
    # technically guaranteed to be non-negative but hedging against floating point issues
    aux = max(b^2 - 4 * a * c, 0)
    # stepsize along dogleg to trust region boundary
    τ = (-b + sqrt(aux)) / (2 * a)

    @bb @. cache.du = cache.u_cauchy + τ * cache.u_cache_2
    return
end

function take_step!(cache::TrustRegionCache)
    @bb copyto!(cache.u_cache, cache.u)
    @bb copyto!(cache.u, cache.u_cache_2)
    @bb copyto!(cache.fu_cache, cache.fu)
    @bb copyto!(cache.fu, cache.fu_cache_2)
end

function not_terminated(cache::TrustRegionCache)
    non_shrink_terminated = cache.force_stop || cache.stats.nsteps ≥ cache.maxiters
    # Terminated due to convergence or maxiters
    non_shrink_terminated && return false
    # Terminated due to too many shrink
    shrink_terminated = cache.shrink_counter ≥ cache.alg.max_shrink_times
    if shrink_terminated
        cache.retcode = ReturnCode.ConvergenceFailure
        return false
    end
    return true
end

# FIXME: Reinit `JᵀJ` operator if `p` is changed
function __reinit_internal!(cache::TrustRegionCache; kwargs...)
    if cache.jvp_operator !== nothing
        cache.jvp_operator = __jacvec(cache.uf, cache.u; cache.fu,
            autodiff = __get_nonsparse_ad(cache.alg.ad))
        @bb cache.Jᵀf = cache.jvp_operator × cache.fu
    end
    cache.loss = __trust_region_loss(cache, cache.fu)
    cache.loss_new = cache.loss
    cache.shrink_counter = 0
    cache.trust_r = convert(eltype(cache.u),
        ifelse(cache.alg.initial_trust_radius == 0, cache.max_trust_r / 11,
            cache.alg.initial_trust_radius))
    cache.make_new_J = true
    return nothing
end

__trust_region_loss(cache::TrustRegionCache, x) = __trust_region_loss(cache.internalnorm, x)
__trust_region_loss(nf::F, x) where {F} = nf(x)^2 / 2

# R-function for adaptive trust region method
function __rfunc(r::R, c2::R, M::R, γ1::R, γ2::R, β::R) where {R <: Real}
    return ifelse(r ≥ c2,
        (2 * (M - 1 - γ2) * atan(r - c2) + (1 + γ2)) / R(π),
        (1 - γ1 - β) * (exp(r - c2) + β / (1 - γ1 - β)))
end
