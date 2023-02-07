"""
```julia
TrustRegion(; chunk_size = Val{0}(), autodiff = Val{true}(),
            standardtag = Val{true}(), concrete_jac = nothing,
            diff_type = Val{:forward}, linsolve = nothing, precs = DEFAULT_PRECS,
            max_trust_radius::Real = 0 // 1,
            initial_trust_radius::Real = 0 // 1,
            step_threshold::Real = 1 // 10,
            shrink_threshold::Real = 1 // 4,
            expand_threshold::Real = 3 // 4,
            shrink_factor::Real = 1 // 4,
            expand_factor::Real = 2 // 1,
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

!!! note

    Currently, the linear solver and chunk size choice only applies to in-place defined
    `NonlinearProblem`s. That is expected to change in the future.
"""
struct TrustRegion{CS, AD, FDT, L, P, ST, CJ, MTR} <:
       AbstractNewtonAlgorithm{CS, AD, FDT, ST, CJ}
    linsolve::L
    precs::P
    max_trust_radius::MTR
    initial_trust_radius::MTR
    step_threshold::MTR
    shrink_threshold::MTR
    expand_threshold::MTR
    shrink_factor::MTR
    expand_factor::MTR
    max_shrink_times::Int
end

function TrustRegion(; chunk_size = Val{0}(),
                     autodiff = Val{true}(),
                     standardtag = Val{true}(), concrete_jac = nothing,
                     diff_type = Val{:forward}, linsolve = nothing, precs = DEFAULT_PRECS,
                     max_trust_radius::Real = 0 // 1,
                     initial_trust_radius::Real = 0 // 1,
                     step_threshold::Real = 1 // 10,
                     shrink_threshold::Real = 1 // 4,
                     expand_threshold::Real = 3 // 4,
                     shrink_factor::Real = 1 // 4,
                     expand_factor::Real = 2 // 1,
                     max_shrink_times::Int = 32)
    TrustRegion{_unwrap_val(chunk_size), _unwrap_val(autodiff), diff_type,
                typeof(linsolve), typeof(precs), _unwrap_val(standardtag),
                _unwrap_val(concrete_jac), typeof(max_trust_radius)
                }(linsolve, precs, max_trust_radius,
                  initial_trust_radius,
                  step_threshold,
                  shrink_threshold,
                  expand_threshold,
                  shrink_factor,
                  expand_factor,
                  max_shrink_times)
end

mutable struct TrustRegionCache{iip, fType, algType, uType, resType, pType,
                                INType, tolType, probType, ufType, L, jType, JC, floatType,
                                trustType, suType, su2Type, tmpType}
    f::fType
    alg::algType
    u::uType
    fu::resType
    p::pType
    uf::ufType
    linsolve::L
    J::jType
    jac_config::JC
    iter::Int
    force_stop::Bool
    maxiters::Int
    internalnorm::INType
    retcode::SciMLBase.ReturnCode.T
    abstol::tolType
    prob::probType
    trust_r::trustType
    max_trust_r::trustType
    step_threshold::suType
    shrink_threshold::trustType
    expand_threshold::trustType
    shrink_factor::trustType
    expand_factor::trustType
    loss::floatType
    loss_new::floatType
    H::jType
    g::resType
    shrink_counter::Int
    step_size::su2Type
    u_tmp::tmpType
    fu_new::resType
    make_new_J::Bool
    r::floatType

    function TrustRegionCache{iip}(f::fType, alg::algType, u::uType, fu::resType, p::pType,
                                   uf::ufType, linsolve::L, J::jType,
                                   jac_config::JC, iter::Int,
                                   force_stop::Bool, maxiters::Int, internalnorm::INType,
                                   retcode::SciMLBase.ReturnCode.T, abstol::tolType,
                                   prob::probType, trust_r::trustType,
                                   max_trust_r::trustType, step_threshold::suType,
                                   shrink_threshold::trustType, expand_threshold::trustType,
                                   shrink_factor::trustType, expand_factor::trustType,
                                   loss::floatType, loss_new::floatType, H::jType,
                                   g::resType, shrink_counter::Int, step_size::su2Type,
                                   u_tmp::tmpType, fu_new::resType, make_new_J::Bool,
                                   r::floatType) where {iip, fType, algType, uType,
                                                        resType, pType, INType,
                                                        tolType, probType, ufType, L,
                                                        jType, JC, floatType, trustType,
                                                        suType, su2Type, tmpType}
        new{iip, fType, algType, uType, resType, pType,
            INType, tolType, probType, ufType, L, jType, JC, floatType,
            trustType, suType, su2Type, tmpType}(f, alg, u, fu, p, uf, linsolve, J,
                                                 jac_config, iter, force_stop,
                                                 maxiters, internalnorm, retcode,
                                                 abstol, prob, trust_r, max_trust_r,
                                                 step_threshold, shrink_threshold,
                                                 expand_threshold, shrink_factor,
                                                 expand_factor, loss,
                                                 loss_new, H, g, shrink_counter,
                                                 step_size, u_tmp, fu_new,
                                                 make_new_J, r)
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
    JacobianWrapper(f, p), nothing, J, zero(u), nothing
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
    uf, linsolve, J, u_tmp, jac_config = jacobian_caches(alg, f, u, p, Val(iip))

    max_trust_radius = convert(eltype(u), alg.max_trust_radius)
    initial_trust_radius = convert(eltype(u), alg.initial_trust_radius)
    step_threshold = convert(eltype(u), alg.step_threshold)
    shrink_threshold = convert(eltype(u), alg.shrink_threshold)
    expand_threshold = convert(eltype(u), alg.expand_threshold)
    shrink_factor = convert(eltype(u), alg.shrink_factor)
    expand_factor = convert(eltype(u), alg.expand_factor)
    # Set default trust region radius if not specified
    if iszero(max_trust_radius)
        max_trust_radius = convert(eltype(u), max(norm(fu), maximum(u) - minimum(u)))
    end
    if iszero(initial_trust_radius)
        initial_trust_radius = convert(eltype(u), max_trust_radius / 11)
    end

    loss_new = loss
    H = ArrayInterfaceCore.undefmatrix(u)
    g = zero(fu)
    shrink_counter = 0
    step_size = zero(u)
    fu_new = zero(fu)
    make_new_J = true
    r = loss

    return TrustRegionCache{iip}(f, alg, u, fu, p, uf, linsolve, J, jac_config,
                                 1, false, maxiters, internalnorm,
                                 ReturnCode.Default, abstol, prob, initial_trust_radius,
                                 max_trust_radius, step_threshold, shrink_threshold,
                                 expand_threshold, shrink_factor, expand_factor, loss,
                                 loss_new, H, g, shrink_counter, step_size, u_tmp, fu_new,
                                 make_new_J, r)
end

function perform_step!(cache::TrustRegionCache{true})
    @unpack make_new_J, J, fu, f, u, p, u_tmp, alg, linsolve = cache
    if cache.make_new_J
        jacobian!(J, cache)
        mul!(cache.H, J, J)
        mul!(cache.g, J, fu)
    end

    linres = dolinsolve(alg.precs, linsolve, A = cache.H, b = _vec(cache.g),
                        linu = _vec(u_tmp),
                        p = p, reltol = cache.abstol)
    cache.linsolve = linres.cache
    cache.u_tmp .= -1 .* u_tmp
    dogleg!(cache)

    # Compute the potentially new u
    cache.u_tmp .= u .+ cache.step_size
    f(cache.fu_new, cache.u_tmp, p)

    trust_region_step!(cache)
    return nothing
end

function perform_step!(cache::TrustRegionCache{false})
    @unpack make_new_J, fu, f, u, p = cache

    if make_new_J
        J = jacobian(cache, f)
        cache.H = J * J
        cache.g = J * fu
    end

    @unpack g, H = cache
    # Compute the Newton step.
    cache.u_tmp = -H \ g
    dogleg!(cache)

    # Compute the potentially new u
    cache.u_tmp = u .+ cache.step_size
    cache.fu_new = f(cache.u_tmp, p)

    trust_region_step!(cache)
    return nothing
end

function trust_region_step!(cache::TrustRegionCache)
    @unpack fu_new, step_size, g, H, loss, max_trust_r = cache
    cache.loss_new = get_loss(fu_new)

    # Compute the ratio of the actual reduction to the predicted reduction.
    cache.r = -(loss - cache.loss_new) / (step_size' * g + step_size' * H * step_size / 2)
    @unpack r = cache

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
end

function dogleg!(cache::TrustRegionCache)
    @unpack u_tmp, trust_r = cache

    # Test if the full step is within the trust region.
    if norm(u_tmp) ≤ trust_r
        cache.step_size = u_tmp
        return
    end

    # Calcualte Cauchy point, optimum along the steepest descent direction.
    δsd = -cache.g
    norm_δsd = norm(δsd)
    if norm_δsd ≥ trust_r
        cache.step_size = δsd .* trust_r / norm_δsd
        return
    end

    # Find the intersection point on the boundary.
    N_sd = u_tmp - δsd
    dot_N_sd = dot(N_sd, N_sd)
    dot_sd_N_sd = dot(δsd, N_sd)
    dot_sd = dot(δsd, δsd)
    fact = dot_sd_N_sd^2 - dot_N_sd * (dot_sd - trust_r^2)
    τ = (-dot_sd_N_sd + sqrt(fact)) / dot_N_sd
    cache.step_size = δsd + τ * N_sd
end

function take_step!(cache::TrustRegionCache{true})
    cache.u .= cache.u_tmp
    cache.fu .= cache.fu_new
end

function take_step!(cache::TrustRegionCache{false})
    cache.u = cache.u_tmp
    cache.fu = cache.fu_new
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
