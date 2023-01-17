"""
```julia
TrustRegion(max_trust_radius::Number; chunk_size = Val{0}(), autodiff = Val{true}(),
                standardtag = Val{true}(), concrete_jac = nothing,
                diff_type = Val{:forward}, linsolve = nothing, precs = DEFAULT_PRECS,
                initial_trust_radius::Number = max_trust_radius / 11,
                step_threshold::Number = 0.1,
                shrink_threshold::Number = 0.25,
                expand_threshold::Number = 0.75,
                shrink_factor::Number = 0.25,
                expand_factor::Number = 2.0,
                max_shrink_times::Int = 32)
```

An advanced TrustRegion implementation with support for efficient handling of sparse
matrices via colored automatic differentiation and preconditioned linear solvers. Designed
for large-scale and numerically-difficult nonlinear systems.

### Keyword Arguments

- `chunk_size`: the chunk size used by the internal ForwardDiff.jl automatic differentiation
  system. This allows for multiple derivative columns to be computed simultaneously,
  improving performance. Defaults to `0`, which is equivalent to using ForwardDiff.jl's
  default chunk size mechanism. For more details, see the documentation for
  [ForwardDiff.jl](https://juliadiff.org/ForwardDiff.jl/stable/).
- `autodiff`: whether to use forward-mode automatic differentiation for the Jacobian.
  Note that this argument is ignored if an analytical Jacobian is passed, as that will be
  used instead. Defaults to `Val{true}`, which means ForwardDiff.jl via
  SparseDiffTools.jl is used by default. If `Val{false}`, then FiniteDiff.jl is used for
  finite differencing.
- `standardtag`: whether to use a standardized tag definition for the purposes of automatic
  differentiation. Defaults to true, which thus uses the `NonlinearSolveTag`. If `Val{false}`,
  then ForwardDiff's default function naming tag is used, which results in larger stack
  traces.
- `concrete_jac`: whether to build a concrete Jacobian. If a Krylov-subspace method is used,
  then the Jacobian will not be constructed and instead direct Jacobian-vector products
  `J*v` are computed using forward-mode automatic differentiation or finite differencing
  tricks (without ever constructing the Jacobian). However, if the Jacobian is still needed,
  for example for a preconditioner, `concrete_jac = true` can be passed in order to force
  the construction of the Jacobian.
- `diff_type`: the type of finite differencing used if `autodiff = false`. Defaults to
  `Val{:forward}` for forward finite differences. For more details on the choices, see the
  [FiniteDiff.jl](https://github.com/JuliaDiff/FiniteDiff.jl) documentation.
- `linsolve`: the [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl) used for the
  linear solves within the Newton method. Defaults to `nothing`, which means it uses the
  LinearSolve.jl default algorithm choice. For more information on available algorithm
  choices, see the [LinearSolve.jl documentation](https://docs.sciml.ai/LinearSolve/stable/).
- `precs`: the choice of preconditioners for the linear solver. Defaults to using no
  preconditioners. For more information on specifying preconditioners for LinearSolve
  algorithms, consult the
  [LinearSolve.jl documentation](https://docs.sciml.ai/LinearSolve/stable/).
- `initial_trust_radius`: the initial trust region radius. Defaults to
  `max_trust_radius / 11`.
- `step_threshold`: the threshold for taking a step. In every iteration, the threshold is
  compared with a value `r`, which is the actual reduction in the objective function divided
  by the predicted reduction. If `step_threshold > r` the model is not a good approximation,
  and the step is rejected. Defaults to `0.1`. For more details, see
  [Trust-region methods](https://optimization.mccormick.northwestern.edu/index.php/Trust-region_methods)
- `shrink_threshold`: the threshold for shrinking the trust region radius. In every
  iteration, the threshold is compared with a value `r` which is the actual reduction in the
  objective function divided by the predicted reduction. If `shrink_threshold > r` the trust
  region radius is shrunk by `shrink_factor`. Defaults to `0.25`. For more details, see
  [Trust-region methods](https://optimization.mccormick.northwestern.edu/index.php/Trust-region_methods)
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

!!! note

    Currently, the linear solver and chunk size choice only applies to in-place defined
    `NonlinearProblem`s. That is expected to change in the future.
"""
struct TrustRegion{CS, AD, FDT, L, P, ST, CJ} <:
       AbstractNewtonAlgorithm{CS, AD, FDT, ST, CJ}
    linsolve::L
    precs::P
    max_trust_radius::Number
    initial_trust_radius::Number
    step_threshold::Number
    shrink_threshold::Number
    expand_threshold::Number
    shrink_factor::Number
    expand_factor::Number
    max_shrink_times::Int
end

function TrustRegion(max_trust_radius::Number; chunk_size = Val{0}(),
                     autodiff = Val{true}(),
                     standardtag = Val{true}(), concrete_jac = nothing,
                     diff_type = Val{:forward}, linsolve = nothing, precs = DEFAULT_PRECS,
                     initial_trust_radius::Number = max_trust_radius / 11,
                     step_threshold::Number = 0.1,
                     shrink_threshold::Number = 0.25,
                     expand_threshold::Number = 0.75,
                     shrink_factor::Number = 0.25,
                     expand_factor::Number = 2.0,
                     max_shrink_times::Int = 32)
    TrustRegion{_unwrap_val(chunk_size), _unwrap_val(autodiff), diff_type,
                typeof(linsolve), typeof(precs), _unwrap_val(standardtag),
                _unwrap_val(concrete_jac)}(linsolve, precs, max_trust_radius::Number,
                                           initial_trust_radius::Number,
                                           step_threshold::Number,
                                           shrink_threshold::Number,
                                           expand_threshold::Number,
                                           shrink_factor::Number,
                                           expand_factor::Number,
                                           max_shrink_times::Int)
end

mutable struct TrustRegionCache{iip, fType, algType, uType, duType, resType, pType,
                                INType, tolType, probType, ufType, L, jType, JC
                                }
    f::fType
    alg::algType
    u::uType
    fu::resType
    p::pType
    uf::ufType
    linsolve::L
    J::jType
    du1::duType
    jac_config::JC
    iter::Int
    force_stop::Bool
    maxiters::Int
    internalnorm::INType
    retcode::SciMLBase.ReturnCode.T
    abstol::tolType
    prob::probType
    trust_r::Number
    loss::Number
    loss_new::Number
    H::jType
    g::resType
    shrink_counter::Int
    step_size::uType
    u_new::uType
    fu_new::resType
    make_new_J::Bool

    function TrustRegionCache{iip}(f::fType, alg::algType, u::uType, fu::resType, p::pType,
                                   uf::ufType, linsolve::L, J::jType, du1::duType,
                                   jac_config::JC, iter::Int,
                                   force_stop::Bool, maxiters::Int, internalnorm::INType,
                                   retcode::SciMLBase.ReturnCode.T, abstol::tolType,
                                   prob::probType, trust_r::Number, loss::Number,
                                   loss_new::Number, H::jType, g::resType,
                                   shrink_counter::Int, step_size::uType, u_new::uType,
                                   fu_new::resType,
                                   make_new_J::Bool) where {iip, fType, algType, uType,
                                                            duType, resType, pType, INType,
                                                            tolType, probType, ufType, L,
                                                            jType, JC}
        new{iip, fType, algType, uType, duType, resType, pType,
            INType, tolType, probType, ufType, L, jType, JC
            }(f, alg, u, fu, p, uf, linsolve, J,
              du1, jac_config, iter, force_stop,
              maxiters, internalnorm, retcode,
              abstol, prob, trust_r, loss,
              loss_new, H, g, shrink_counter,
              step_size, u_new, fu_new,
              make_new_J)
    end
end

function jacobian_caches(alg::TrustRegion, f, u, p, ::Val{true})
    uf = JacobianWrapper(f, p)
    J = ArrayInterfaceCore.undefmatrix(u)

    linprob = LinearProblem(J, _vec(zero(u)); u0 = _vec(zero(u)))
    weight = similar(u)
    recursivefill!(weight, false)

    Pl, Pr = wrapprecs(alg.precs(J, nothing, u, p, nothing, nothing, nothing, nothing,
                                 nothing)..., weight)
    linsolve = init(linprob, alg.linsolve, alias_A = true, alias_b = true,
                    Pl = Pl, Pr = Pr)

    du1 = zero(u)
    du2 = zero(u)
    tmp = zero(u)
    jac_config = build_jac_config(alg, f, uf, du1, u, tmp, du2)

    uf, linsolve, J, du1, jac_config
end

function jacobian_caches(alg::TrustRegion, f, u, p, ::Val{false})
    J = ArrayInterfaceCore.undefmatrix(u)
    JacobianWrapper(f, p), nothing, J, nothing, nothing
end

function SciMLBase.__init(prob::NonlinearProblem{uType, iip}, alg::TrustRegion,
                          args...;
                          alias_u0 = false,
                          maxiters = 1000,
                          abstol = 1e-6,
                          internalnorm = DEFAULT_NORM,
                          kwargs...) where {uType, iip}
    if alias_u0
        u = prob.u0
    else
        u = deepcopy(prob.u0)
    end
    f = prob.f
    p = prob.p
    if iip
        fu = zero(u)
        f(fu, u, p)
    else
        fu = f(u, p)
    end

    loss = get_loss(fu)
    uf, linsolve, J, du1, jac_config = jacobian_caches(alg, f, u, p, Val(iip))

    return TrustRegionCache{iip}(f, alg, u, fu, p, uf, linsolve, J, du1, jac_config,
                                 1, false, maxiters, internalnorm,
                                 ReturnCode.Default, abstol, prob, alg.initial_trust_radius,
                                 loss, loss, J, fu, 0, u, u, fu, true)
end

function perform_step!(cache::TrustRegionCache{true})
    @unpack make_new_J, J, fu, f, u, p = cache
    if cache.make_new_J
        jacobian!(J, cache)
        cache.H = J * J
        cache.g = J * fu
    end

    dogleg!(cache)

    # Compute the potentially new u
    cache.u_new = u .+ cache.step_size
    f(cache.fu_new, cache.u_new, p)

    trust_region_step!(cache)
    return nothing
end

function perform_step!(cache::TrustRegionCache{false})
    @unpack make_new_J, fu, f, u, p = cache

    if make_new_J
        J = jacobian(cache, f)
        mul!(cache.H,J,J)
        mul!(cache.g,J,fu)
    end
    dogleg!(cache)

    # Compute the potentially new u
    cache.u_new = u .+ cache.step_size
    cache.fu_new = f(cache.u_new, p)

    trust_region_step!(cache)
    return nothing
end

function trust_region_step!(cache::TrustRegionCache)
    @unpack fu_new, u_new, step_size, g, H, loss, alg = cache
    cache.loss_new = get_loss(fu_new)

    # Compute the ratio of the actual reduction to the predicted reduction.
    model = -(step_size' * g + 0.5 * step_size' * H * step_size)
    r = (loss - cache.loss_new) / model

    # Update the trust region radius.
    if r < alg.shrink_threshold
        cache.trust_r *= alg.shrink_factor
        cache.shrink_counter += 1
    else
        cache.shrink_counter = 0
    end
    if r > alg.step_threshold

        # Take the step.
        cache.u = u_new
        cache.fu = fu_new
        cache.loss = cache.loss_new

        # Update the trust region radius.
        if r > alg.expand_threshold
            cache.trust_r = min(alg.expand_factor * cache.trust_r, alg.max_trust_radius)
        end

        cache.make_new_J = true
    else
        # No need to make a new J, no step was taken, so we try again with a smaller trust_r
        cache.make_new_J = false
    end

    if iszero(cache.fu) || cache.internalnorm(cache.step_size) < cache.abstol
        cache.force_stop = true
    end
end

function dogleg!(cache::TrustRegionCache)
    @unpack g, H, trust_r = cache
    # Compute the Newton step.
    δN = -H \ g
    # Test if the full step is within the trust region.
    if norm(δN) ≤ trust_r
        cache.step_size = δN
        return
    end

    # Calcualte Cauchy point, optimum along the steepest descent direction.
    δsd = -g
    norm_δsd = norm(δsd)
    if norm_δsd ≥ trust_r
        cache.step_size = δsd .* trust_r / norm_δsd
        return
    end

    # Find the intersection point on the boundary.
    N_sd = δN - δsd
    dot_N_sd = dot(N_sd, N_sd)
    dot_sd_N_sd = dot(δsd, N_sd)
    dot_sd = dot(δsd, δsd)
    fact = dot_sd_N_sd^2 - dot_N_sd * (dot_sd - trust_r^2)
    τ = (-dot_sd_N_sd + sqrt(fact)) / dot_N_sd
    cache.step_size = δsd + τ * N_sd
end

function SciMLBase.solve!(cache::TrustRegionCache)
    while !cache.force_stop && cache.iter < cache.maxiters &&
              cache.shrink_counter < cache.alg.max_shrink_times
        perform_step!(cache)
        cache.iter += 1
    end

    if cache.iter == cache.maxiters
        cache.retcode = ReturnCode.MaxIters
    else
        cache.retcode = ReturnCode.Success
    end

    SciMLBase.build_solution(cache.prob, cache.alg, cache.u, cache.fu;
                             retcode = cache.retcode)
end
