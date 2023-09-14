"""
`RadiusUpdateSchemes`

`RadiusUpdateSchemes` is the standard enum interface for different types of radius update schemes
implemented in the Trust Region method. These schemes specify how the radius of the so-called trust region
is updated after each iteration of the algorithm. The specific role and caveats associated with each
scheme are provided below.

## Using `RadiusUpdateSchemes`

`RadiusUpdateSchemes` uses the standard EnumX interface (https://github.com/fredrikekre/EnumX.jl),
and hence inherits all properties of being an EnumX, including the type of each constituent enum
states as `RadiusUpdateSchemes.T`. Simply put the desired scheme as follows:
`TrustRegion(radius_update_scheme = your desired update scheme)`. For example,
`sol = solve(prob, alg=TrustRegion(radius_update_scheme = RadiusUpdateSchemes.Hei))`.
"""
@enumx RadiusUpdateSchemes begin
    """
    `RadiusUpdateSchemes.Simple`

    The simple or conventional radius update scheme. This scheme is chosen by default
    and follows the conventional approach to update the trust region radius, i.e. if the
    trial step is accepted it increases the radius by a fixed factor (bounded by a maximum radius)
    and if the trial step is rejected, it shrinks the radius by a fixed factor.
    """
    Simple

    """
    `RadiusUpdateSchemes.NLsolve`

    The same updating rule as in NLsolve's trust region implementation
    """
    NLsolve

    """
    `RadiusUpdateSchemes.Hei`

    This scheme is proposed by [Hei, L.] (https://www.jstor.org/stable/43693061). The trust region radius
    depends on the size (norm) of the current step size. The hypothesis is to let the radius converge to zero
    as the iterations progress, which is more reliable and robust for ill-conditioned as well as degenerate
    problems.
    """
    Hei

    """
    `RadiusUpdateSchemes.Yuan`

    This scheme is proposed by [Yuan, Y.] (https://www.researchgate.net/publication/249011466_A_new_trust_region_algorithm_with_trust_region_radius_converging_to_zero).
    Similar to Hei's scheme, the trust region is updated in a way so that it converges to zero, however here,
    the radius depends on the size (norm) of the current gradient of the objective (merit) function. The hypothesis
    is that the step size is bounded by the gradient size, so it makes sense to let the radius depend on the gradient.
    """
    Yuan

    """
    `RadiusUpdateSchemes.Bastin`

    This scheme is proposed by [Bastin, et al.] (https://www.researchgate.net/publication/225100660_A_retrospective_trust-region_method_for_unconstrained_optimization).
    The scheme is called a retrospective update scheme as it uses the model function at the current
    iteration to compute the ratio of the actual reduction and the predicted reduction in the previous
    trial step, and use this ratio to update the trust region radius. The hypothesis is to exploit the information
    made available during the optimization process in order to vary the accuracy of the objective function computation.
    """
    Bastin

    """
    `RadiusUpdateSchemes.Fan`

    This scheme is proposed by [Fan, J.] (https://link.springer.com/article/10.1007/s10589-005-3078-8). It is very much similar to
    Hei's and Yuan's schemes as it lets the trust region radius depend on the current size (norm) of the objective (merit)
    function itself. These new update schemes are known to improve local convergence.
    """
    Fan
end

"""
```julia
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
    `AutoForwardDiff()`. Valid choices are types from ADTypes.jl.
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
end

function TrustRegion(; concrete_jac = nothing, linsolve = nothing, precs = DEFAULT_PRECS,
    radius_update_scheme::RadiusUpdateSchemes.T = RadiusUpdateSchemes.Simple, #defaults to conventional radius update
    max_trust_radius::Real = 0 // 1, initial_trust_radius::Real = 0 // 1,
    step_threshold::Real = 1 // 10, shrink_threshold::Real = 1 // 4,
    expand_threshold::Real = 3 // 4, shrink_factor::Real = 1 // 4,
    expand_factor::Real = 2 // 1, max_shrink_times::Int = 32, adkwargs...)
    ad = default_adargs_to_adtype(; adkwargs...)
    return TrustRegion{_unwrap_val(concrete_jac)}(ad, linsolve, precs, radius_update_scheme,
        max_trust_radius, initial_trust_radius, step_threshold, shrink_threshold,
        expand_threshold, shrink_factor, expand_factor, max_shrink_times)
end

@concrete mutable struct TrustRegionCache{iip, trustType, floatType}
    f
    alg
    u_prev
    u
    fu_prev
    fu
    fu2
    p
    uf
    linsolve
    J
    jac_cache
    force_stop::Bool
    maxiters::Int
    internalnorm
    retcode::ReturnCode.T
    abstol
    prob
    radius_update_scheme::RadiusUpdateSchemes.T
    trust_r::trustType
    max_trust_r::trustType
    step_threshold
    shrink_threshold::trustType
    expand_threshold::trustType
    shrink_factor::trustType
    expand_factor::trustType
    loss::floatType
    loss_new::floatType
    H
    g
    shrink_counter::Int
    step_size
    u_tmp
    fu_new
    make_new_J::Bool
    r::floatType
    p1::floatType
    p2::floatType
    p3::floatType
    p4::floatType
    ϵ::floatType
    stats::NLStats
end

function SciMLBase.__init(prob::NonlinearProblem{uType, iip}, alg::TrustRegion, args...;
    alias_u0 = false, maxiters = 1000, abstol = 1e-8, internalnorm = DEFAULT_NORM,
    linsolve_kwargs = (;), kwargs...) where {uType, iip}
    @unpack f, u0, p = prob
    u = alias_u0 ? u0 : deepcopy(u0)
    u_prev = zero(u)
    fu1 = evaluate_f(prob, u)
    fu_prev = zero(fu1)

    loss = get_loss(fu1)
    uf, linsolve, J, fu2, jac_cache, du = jacobian_caches(alg, f, u, p, Val(iip);
        linsolve_kwargs)

    radius_update_scheme = alg.radius_update_scheme
    max_trust_radius = convert(eltype(u), alg.max_trust_radius)
    initial_trust_radius = convert(eltype(u), alg.initial_trust_radius)
    step_threshold = convert(eltype(u), alg.step_threshold)
    shrink_threshold = convert(eltype(u), alg.shrink_threshold)
    expand_threshold = convert(eltype(u), alg.expand_threshold)
    shrink_factor = convert(eltype(u), alg.shrink_factor)
    expand_factor = convert(eltype(u), alg.expand_factor)
    # Set default trust region radius if not specified
    if iszero(max_trust_radius)
        max_trust_radius = convert(eltype(u), max(norm(fu1), maximum(u) - minimum(u)))
    end
    if iszero(initial_trust_radius)
        initial_trust_radius = convert(eltype(u), max_trust_radius / 11)
    end

    loss_new = loss
    H = zero(J)
    g = _mutable_zero(fu1)
    shrink_counter = 0
    step_size = zero(u)
    fu_new = zero(fu1)
    make_new_J = true
    r = loss

    # Parameters for the Schemes
    p1 = convert(eltype(u), 0.0)
    p2 = convert(eltype(u), 0.0)
    p3 = convert(eltype(u), 0.0)
    p4 = convert(eltype(u), 0.0)
    ϵ = convert(eltype(u), 1.0e-8)
    if radius_update_scheme === RadiusUpdateSchemes.NLsolve
        p1 = convert(eltype(u), 0.5)
    elseif radius_update_scheme === RadiusUpdateSchemes.Hei
        step_threshold = convert(eltype(u), 0.0)
        shrink_threshold = convert(eltype(u), 0.25)
        expand_threshold = convert(eltype(u), 0.25)
        p1 = convert(eltype(u), 5.0) # M
        p2 = convert(eltype(u), 0.1) # β
        p3 = convert(eltype(u), 0.15) # γ1
        p4 = convert(eltype(u), 0.15) # γ2
        initial_trust_radius = convert(eltype(u), 1.0)
    elseif radius_update_scheme === RadiusUpdateSchemes.Yuan
        step_threshold = convert(eltype(u), 0.0001)
        shrink_threshold = convert(eltype(u), 0.25)
        expand_threshold = convert(eltype(u), 0.25)
        p1 = convert(eltype(u), 2.0) # μ
        p2 = convert(eltype(u), 1 / 6) # c5
        p3 = convert(eltype(u), 6.0) # c6
        p4 = convert(eltype(u), 0.0)
        if iip
            auto_jacvec!(g, (fu, x) -> f(fu, x, p), u, fu1)
        else
            if isa(u, Number)
                g = ForwardDiff.derivative(x -> f(x, p), u)
            else
                g = auto_jacvec(x -> f(x, p), u, fu1)
            end
        end
        initial_trust_radius = convert(eltype(u), p1 * norm(g))
    elseif radius_update_scheme === RadiusUpdateSchemes.Fan
        step_threshold = convert(eltype(u), 0.0001)
        shrink_threshold = convert(eltype(u), 0.25)
        expand_threshold = convert(eltype(u), 0.75)
        p1 = convert(eltype(u), 0.1) # μ
        p2 = convert(eltype(u), 1 / 4) # c5
        p3 = convert(eltype(u), 12) # c6
        p4 = convert(eltype(u), 1.0e18) # M
        initial_trust_radius = convert(eltype(u), p1 * (norm(fu1)^0.99))
    elseif radius_update_scheme === RadiusUpdateSchemes.Bastin
        step_threshold = convert(eltype(u), 0.05)
        shrink_threshold = convert(eltype(u), 0.05)
        expand_threshold = convert(eltype(u), 0.9)
        p1 = convert(eltype(u), 2.5)  #alpha_1
        p2 = convert(eltype(u), 0.25) # alpha_2
        p3 = convert(eltype(u), 0) # not required
        p4 = convert(eltype(u), 0) # not required
        initial_trust_radius = convert(eltype(u), 1.0)
    end

    return TrustRegionCache{iip}(f, alg, u_prev, u, fu_prev, fu1, fu2, p, uf, linsolve, J,
        jac_cache, false, maxiters, internalnorm, ReturnCode.Default, abstol, prob,
        radius_update_scheme, initial_trust_radius, max_trust_radius, step_threshold,
        shrink_threshold, expand_threshold, shrink_factor, expand_factor, loss, loss_new,
        H, g, shrink_counter, step_size, du, fu_new, make_new_J, r, p1, p2, p3, p4, ϵ,
        NLStats(1, 0, 0, 0, 0))
end

isinplace(::TrustRegionCache{iip}) where {iip} = iip

function perform_step!(cache::TrustRegionCache{true})
    @unpack make_new_J, J, fu, f, u, p, u_tmp, alg, linsolve = cache
    if cache.make_new_J
        jacobian!(J, cache)
        mul!(cache.H, J', J)
        mul!(cache.g, J', fu)
        cache.stats.njacs += 1
    end

    linres = dolinsolve(alg.precs, linsolve, A = J, b = _vec(fu), # cache.H, b = _vec(cache.g),
        linu = _vec(u_tmp),
        p = p, reltol = cache.abstol)
    cache.linsolve = linres.cache
    cache.u_tmp .= -1 .* u_tmp
    dogleg!(cache)

    # Compute the potentially new u
    cache.u_tmp .= u .+ cache.step_size
    f(cache.fu_new, cache.u_tmp, p)
    trust_region_step!(cache)
    cache.stats.nf += 1
    cache.stats.nsolve += 1
    cache.stats.nfactors += 1
    return nothing
end

function perform_step!(cache::TrustRegionCache{false})
    @unpack make_new_J, fu, f, u, p = cache

    if make_new_J
        J = jacobian(cache, f)
        cache.H = J' * J
        cache.g = J' * fu
        cache.stats.njacs += 1
    end

    @unpack g, H = cache
    # Compute the Newton step.
    cache.u_tmp = -H \ g
    dogleg!(cache)

    # Compute the potentially new u
    cache.u_tmp = u .+ cache.step_size
    cache.fu_new = f(cache.u_tmp, p)
    trust_region_step!(cache)
    cache.stats.nf += 1
    cache.stats.nsolve += 1
    cache.stats.nfactors += 1
    return nothing
end

function retrospective_step!(cache::TrustRegionCache)
    @unpack J, fu_prev, fu, u_prev, u = cache
    J = jacobian!!(deepcopy(J), cache)
    if J isa Number
        cache.H = J * J
        cache.g = J * fu
    else
        mul!(cache.H, J, J)
        mul!(cache.g, J, fu)
    end
    cache.stats.njacs += 1
    @unpack H, g, step_size = cache

    return -(get_loss(fu_prev) - get_loss(fu)) /
           (dot(step_size, g) + dot(step_size, H, step_size) / 2)
end

function trust_region_step!(cache::TrustRegionCache)
    @unpack fu_new, step_size, g, H, loss, max_trust_r, radius_update_scheme = cache
    cache.loss_new = get_loss(fu_new)

    # Compute the ratio of the actual reduction to the predicted reduction.
    cache.r = -(loss - cache.loss_new) / (dot(step_size, g) + dot(step_size, H, step_size) / 2)
    @unpack r = cache 

    if radius_update_scheme === RadiusUpdateSchemes.Simple
        # Update the trust region radius.
        if r < cache.shrink_threshold
            cache.trust_r *= cache.shrink_factor
            cache.shrink_counter += 1
        else
            cache.shrink_counter = 0
        end
        if r > cache.step_threshold
            take_step!(cache)
            cache.loss = cache.loss_new

            # Update the trust region radius.
            if r > cache.expand_threshold
                cache.trust_r = min(cache.expand_factor * cache.trust_r, max_trust_r)
            end

            cache.make_new_J = true
        else
            # No need to make a new J, no step was taken, so we try again with a smaller trust_r
            cache.make_new_J = false
        end

        if iszero(cache.fu) || cache.internalnorm(cache.fu) < cache.abstol
            cache.force_stop = true
        end

    elseif radius_update_scheme === RadiusUpdateSchemes.NLsolve
        # accept/reject decision
        if r > cache.step_threshold # accept
            take_step!(cache)
            cache.loss = cache.loss_new
            cache.make_new_J = true
        else # reject
            cache.make_new_J = false
        end

        # trust region update 
        if r < cache.shrink_threshold # default 1 // 10
            cache.trust_r *= cache.shrink_factor # default 1 // 2
        elseif r >= cache.expand_threshold # default 9 // 10
            cache.trust_r = cache.expand_factor * norm(cache.step_size) # default 2
        elseif r >= cache.p1 # default 1 // 2
            cache.trust_r = max(cache.trust_r, cache.expand_factor * norm(cache.step_size))
        end

        # convergence test
        if iszero(cache.fu) || cache.internalnorm(cache.fu) < cache.abstol
            cache.force_stop = true
        end

    elseif radius_update_scheme === RadiusUpdateSchemes.Hei
        if r > cache.step_threshold
            take_step!(cache)
            cache.loss = cache.loss_new
            cache.make_new_J = true
        else
            cache.make_new_J = false
        end
        # Hei's radius update scheme
        @unpack shrink_threshold, p1, p2, p3, p4 = cache
        if rfunc(r, shrink_threshold, p1, p3, p4, p2) * cache.internalnorm(step_size) <
           cache.trust_r
            cache.shrink_counter += 1
        else
            cache.shrink_counter = 0
        end
        cache.trust_r = rfunc(r, shrink_threshold, p1, p3, p4, p2) *
                        cache.internalnorm(step_size)

        if iszero(cache.fu) || cache.internalnorm(cache.fu) < cache.abstol ||
           cache.internalnorm(g) < cache.ϵ
            cache.force_stop = true
        end

    elseif radius_update_scheme === RadiusUpdateSchemes.Yuan
        if r < cache.shrink_threshold
            cache.p1 = cache.p2 * cache.p1
            cache.shrink_counter += 1
        elseif r >= cache.expand_threshold &&
               cache.internalnorm(step_size) > cache.trust_r / 2
            cache.p1 = cache.p3 * cache.p1
            cache.shrink_counter = 0
        end

        if r > cache.step_threshold
            take_step!(cache)
            cache.loss = cache.loss_new
            cache.make_new_J = true
        else
            cache.make_new_J = false
        end

        @unpack p1 = cache
        cache.trust_r = p1 * cache.internalnorm(jvp!(cache))
        if iszero(cache.fu) || cache.internalnorm(cache.fu) < cache.abstol ||
           cache.internalnorm(g) < cache.ϵ
            cache.force_stop = true
        end
        #Fan's update scheme
    elseif radius_update_scheme === RadiusUpdateSchemes.Fan
        if r < cache.shrink_threshold
            cache.p1 *= cache.p2
            cache.shrink_counter += 1
        elseif r > cache.expand_threshold
            cache.p1 = min(cache.p1 * cache.p3, cache.p4)
            cache.shrink_counter = 0
        end

        if r > cache.step_threshold
            take_step!(cache)
            cache.loss = cache.loss_new
            cache.make_new_J = true
        else
            cache.make_new_J = false
        end

        @unpack p1 = cache
        cache.trust_r = p1 * (cache.internalnorm(cache.fu)^0.99)
        if iszero(cache.fu) || cache.internalnorm(cache.fu) < cache.abstol ||
           cache.internalnorm(g) < cache.ϵ
            cache.force_stop = true
        end
    elseif radius_update_scheme === RadiusUpdateSchemes.Bastin
        if r > cache.step_threshold
            take_step!(cache)
            cache.loss = cache.loss_new
            cache.make_new_J = true
            if retrospective_step!(cache) >= cache.expand_threshold
                cache.trust_r = max(cache.p1 * cache.internalnorm(step_size), cache.trust_r)
            end

        else
            cache.make_new_J = false
            cache.trust_r *= cache.p2
            cache.shrink_counter += 1
        end
        if iszero(cache.fu) || cache.internalnorm(cache.fu) < cache.abstol
            cache.force_stop = true
        end
    end
end

function dogleg!(cache::TrustRegionCache)
    @unpack u_tmp, trust_r = cache

    # Test if the full step is within the trust region.
    if norm(u_tmp) ≤ trust_r
        cache.step_size = deepcopy(u_tmp)
        return
    end

    # Calcualte Cauchy point, optimum along the steepest descent direction.
    l_grad = norm(cache.g)
    d_cauchy = l_grad^3 / dot(cache.g, cache.H, cache.g) # distance of the cauchy point from the current iterate
    if d_cauchy > trust_r # cauchy point lies outside of trust region
        cache.step_size = - (trust_r/l_grad) * cache.g # step to the end of the trust region
        return
    end
    
    # cauchy point lies inside the trust region
    u_c = - (d_cauchy/l_grad) * cache.g

    # Find the intersection point on the boundary.
    Δ = u_tmp - u_c # calf of the dogleg
    θ = dot(Δ, u_c) # ~ cos(∠(calf,thigh)) 
    l_calf = dot(Δ,Δ) # length of the calf of the dogleg
    aux = max(θ^2 + l_calf*(trust_r^2 - d_cauchy^2), 0) # technically guaranteed to be non-negative but hedging against floating point issues
    τ = ( - θ + sqrt( aux ) ) / l_calf # stepsize along dogleg
    cache.step_size = u_c + τ * Δ
end

function take_step!(cache::TrustRegionCache{true})
    cache.u_prev .= cache.u
    cache.u .= cache.u_tmp
    cache.fu_prev .= cache.fu
    cache.fu .= cache.fu_new
end

function take_step!(cache::TrustRegionCache{false})
    cache.u_prev = cache.u
    cache.u = cache.u_tmp
    cache.fu_prev = cache.fu
    cache.fu = cache.fu_new
end

function jvp!(cache::TrustRegionCache{false})
    @unpack f, u, fu, uf = cache
    if isa(u, Number)
        return value_derivative(uf, u)
    end
    return auto_jacvec(uf, u, fu)
end

function jvp!(cache::TrustRegionCache{true})
    @unpack g, f, u, fu, uf = cache
    if isa(u, Number)
        return value_derivative(uf, u)
    end
    auto_jacvec!(g, uf, u, fu)
    return g
end

function SciMLBase.solve!(cache::TrustRegionCache)
    while !cache.force_stop && cache.stats.nsteps < cache.maxiters &&
              cache.shrink_counter < cache.alg.max_shrink_times
        perform_step!(cache)
        cache.stats.nsteps += 1
    end

    if cache.stats.nsteps == cache.maxiters
        cache.retcode = ReturnCode.MaxIters
    else
        cache.retcode = ReturnCode.Success
    end

    return SciMLBase.build_solution(cache.prob, cache.alg, cache.u, cache.fu; cache.retcode,
        cache.stats)
end

function SciMLBase.reinit!(cache::TrustRegionCache{iip}, u0 = cache.u; p = cache.p,
    abstol = cache.abstol, maxiters = cache.maxiters) where {iip}
    cache.p = p
    if iip
        recursivecopy!(cache.u, u0)
        cache.f(cache.fu, cache.u, p)
    else
        # don't have alias_u0 but cache.u is never mutated for OOP problems so it doesn't matter
        cache.u = u0
        cache.fu = cache.f(cache.u, p)
    end
    cache.abstol = abstol
    cache.maxiters = maxiters
    cache.stats.nf = 1
    cache.stats.nsteps = 1
    cache.force_stop = false
    cache.retcode = ReturnCode.Default
    cache.make_new_J = true
    cache.loss = get_loss(cache.fu)
    cache.shrink_counter = 0
    cache.trust_r = convert(eltype(cache.u), cache.alg.initial_trust_radius)
    if iszero(cache.trust_r)
        cache.trust_r = convert(eltype(cache.u), cache.max_trust_r / 11)
    end
    return cache
end
