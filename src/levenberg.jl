"""
    LevenbergMarquardt(; concrete_jac = nothing, linsolve = nothing,
        precs = DEFAULT_PRECS, damping_initial::Real = 1.0,
        damping_increase_factor::Real = 2.0, damping_decrease_factor::Real = 3.0,
        finite_diff_step_geodesic::Real = 0.1, α_geodesic::Real = 0.75,
        b_uphill::Real = 1.0, min_damping_D::AbstractFloat = 1e-8, adkwargs...)

An advanced Levenberg-Marquardt implementation with the improvements suggested in the
[paper](https://arxiv.org/abs/1201.5885) "Improvements to the Levenberg-Marquardt
algorithm for nonlinear least-squares minimization". Designed for large-scale and
numerically-difficult nonlinear systems.

If no `linsolve` is provided or a variant of `QR` is provided, then we will use an efficient
routine for the factorization without constructing `JᵀJ` and `Jᵀf`. For more details see
"Chapter 10: Implementation of the Levenberg-Marquardt Method" of
["Numerical Optimization" by Jorge Nocedal & Stephen J. Wright](https://link.springer.com/book/10.1007/978-0-387-40065-5).

### Keyword Arguments

  - `autodiff`: determines the backend used for the Jacobian. Note that this argument is
    ignored if an analytical Jacobian is passed, as that will be used instead. Defaults to
    `nothing` which means that a default is selected according to the problem specification!
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
  - `damping_initial`: the starting value for the damping factor. The damping factor is
    inversely proportional to the step size. The damping factor is adjusted during each
    iteration. Defaults to `1.0`. For more details, see section 2.1 of
    [this paper](https://arxiv.org/abs/1201.5885).
  - `damping_increase_factor`: the factor by which the damping is increased if a step is
    rejected. Defaults to `2.0`. For more details, see section 2.1 of
    [this paper](https://arxiv.org/abs/1201.5885).
  - `damping_decrease_factor`: the factor by which the damping is decreased if a step is
    accepted. Defaults to `3.0`. For more details, see section 2.1 of
    [this paper](https://arxiv.org/abs/1201.5885).
  - `finite_diff_step_geodesic`: the step size used for finite differencing used to calculate
    the geodesic acceleration. Defaults to `0.1` which means that the step size is
    approximately 10% of the first-order step. For more details, see section 3 of
    [this paper](https://arxiv.org/abs/1201.5885).
  - `α_geodesic`: a factor that determines if a step is accepted or rejected. To incorporate
    geodesic acceleration as an addition to the Levenberg-Marquardt algorithm, it is necessary
    that acceptable steps meet the condition
    ``\\frac{2||a||}{||v||} \\le \\alpha_{\\text{geodesic}}``, where ``a`` is the geodesic
    acceleration, ``v`` is the Levenberg-Marquardt algorithm's step (velocity along a geodesic
    path) and `α_geodesic` is some number of order `1`. For most problems `α_geodesic = 0.75`
    is a good value but for problems where convergence is difficult `α_geodesic = 0.1` is an
    effective choice. Defaults to `0.75`. For more details, see section 3, equation (15) of
    [this paper](https://arxiv.org/abs/1201.5885).
  - `b_uphill`: a factor that determines if a step is accepted or rejected. The standard
    choice in the Levenberg-Marquardt method is to accept all steps that decrease the cost
    and reject all steps that increase the cost. Although this is a natural and safe choice,
    it is often not the most efficient. Therefore downhill moves are always accepted, but
    uphill moves are only conditionally accepted. To decide whether an uphill move will be
    accepted at each iteration ``i``, we compute
    ``\\beta_i = \\cos(v_{\\text{new}}, v_{\\text{old}})``, which denotes the cosine angle
    between the proposed velocity ``v_{\\text{new}}`` and the velocity of the last accepted
    step ``v_{\\text{old}}``. The idea is to accept uphill moves if the angle is small. To
    specify, uphill moves are accepted if
    ``(1-\\beta_i)^{b_{\\text{uphill}}} C_{i+1} \\le C_i``, where ``C_i`` is the cost at
    iteration ``i``. Reasonable choices for `b_uphill` are `1.0` or `2.0`, with `b_uphill=2.0`
    allowing higher uphill moves than `b_uphill=1.0`. When `b_uphill=0.0`, no uphill moves
    will be accepted. Defaults to `1.0`. For more details, see section 4 of
    [this paper](https://arxiv.org/abs/1201.5885).
  - `min_damping_D`: the minimum value of the damping terms in the diagonal damping matrix
    `DᵀD`, where `DᵀD` is given by the largest diagonal entries of `JᵀJ` yet encountered,
    where `J` is the Jacobian. It is suggested by
    [this paper](https://arxiv.org/abs/1201.5885) to use a minimum value of the elements in
    `DᵀD` to prevent the damping from being too small. Defaults to `1e-8`.
"""
@concrete struct LevenbergMarquardt{CJ, AD} <: AbstractNewtonAlgorithm{CJ, AD}
    ad::AD
    linsolve
    precs
    damping_initial
    damping_increase_factor
    damping_decrease_factor
    finite_diff_step_geodesic
    α_geodesic
    b_uphill
    min_damping_D
end

function set_ad(alg::LevenbergMarquardt{CJ}, ad) where {CJ}
    return LevenbergMarquardt{CJ}(ad, alg.linsolve, alg.precs, alg.damping_initial,
        alg.damping_increase_factor, alg.damping_decrease_factor,
        alg.finite_diff_step_geodesic, alg.α_geodesic, alg.b_uphill, alg.min_damping_D)
end

function LevenbergMarquardt(; concrete_jac = nothing, linsolve = nothing,
        precs = DEFAULT_PRECS, damping_initial::Real = 1.0, α_geodesic::Real = 0.75,
        damping_increase_factor::Real = 2.0, damping_decrease_factor::Real = 3.0,
        finite_diff_step_geodesic::Real = 0.1, b_uphill::Real = 1.0,
        min_damping_D::Real = 1e-8, adkwargs...)
    ad = default_adargs_to_adtype(; adkwargs...)
    _concrete_jac = ifelse(concrete_jac === nothing, true, concrete_jac)
    return LevenbergMarquardt{_unwrap_val(_concrete_jac)}(ad, linsolve, precs,
        damping_initial, damping_increase_factor, damping_decrease_factor,
        finite_diff_step_geodesic, α_geodesic, b_uphill, min_damping_D)
end

@concrete mutable struct LevenbergMarquardtCache{iip, fastls} <:
                         AbstractNonlinearSolveCache{iip}
    f
    alg
    u
    u_cache
    u_cache_2
    fu
    fu_cache
    fu_cache_2
    du
    du_cache
    J
    JᵀJ
    Jv
    DᵀD
    v
    v_cache
    a
    mat_tmp
    rhs_tmp
    p
    uf
    linsolve
    jac_cache
    force_stop::Bool
    maxiters::Int
    internalnorm
    retcode::ReturnCode.T
    abstol
    reltol
    prob
    λ
    λ_factor
    damping_increase_factor
    damping_decrease_factor
    h
    α_geodesic
    b_uphill
    min_damping_D
    norm_v_old
    loss_old
    make_new_J::Bool
    stats::NLStats
    tc_cache_1
    tc_cache_2
    trace
end

function SciMLBase.__init(prob::Union{NonlinearProblem{uType, iip},
            NonlinearLeastSquaresProblem{uType, iip}}, alg_::LevenbergMarquardt,
        args...; alias_u0 = false, maxiters = 1000, abstol = nothing, reltol = nothing,
        termination_condition = nothing, internalnorm::F = DEFAULT_NORM,
        linsolve_kwargs = (;), kwargs...) where {uType, iip, F}
    alg = get_concrete_algorithm(alg_, prob)
    @unpack f, u0, p = prob

    u = __maybe_unaliased(u0, alias_u0)
    T = eltype(u)
    fu = evaluate_f(prob, u)

    fastls = !__needs_square_A(alg, u0)

    if !fastls
        uf, linsolve, J, fu_cache, jac_cache, du, JᵀJ, v = jacobian_caches(alg, f, u, p,
            Val(iip); linsolve_kwargs, linsolve_with_JᵀJ = Val(true))
    else
        uf, linsolve, J, fu_cache, jac_cache, du = jacobian_caches(alg, f, u, p,
            Val(iip); linsolve_kwargs, linsolve_with_JᵀJ = Val(false))
        u_ = _vec(u)
        @bb JᵀJ = similar(u_)
        @bb v = similar(du)
    end

    λ = T(alg.damping_initial)
    λ_factor = T(alg.damping_increase_factor)
    damping_increase_factor = T(alg.damping_increase_factor)
    damping_decrease_factor = T(alg.damping_decrease_factor)
    h = T(alg.finite_diff_step_geodesic)
    α_geodesic = T(alg.α_geodesic)
    b_uphill = T(alg.b_uphill)
    min_damping_D = T(alg.min_damping_D)

    DᵀD = __init_diagonal(u, min_damping_D)

    loss = internalnorm(fu)

    @bb a = similar(du)
    @bb v_old = copy(v)
    @bb δ = similar(du)

    make_new_J = true

    abstol, reltol, tc_cache_1 = init_termination_cache(abstol, reltol, fu, u,
        termination_condition)
    if prob isa NonlinearLeastSquaresProblem
        _, _, tc_cache_2 = init_termination_cache(abstol, reltol, fu, u,
            termination_condition)
    else
        tc_cache_2 = nothing
    end

    trace = init_nonlinearsolve_trace(alg, u, fu, ApplyArray(__zero, J), du; kwargs...)

    if !fastls
        @bb mat_tmp = similar(JᵀJ)
        @bb mat_tmp .*= T(0)
        rhs_tmp = nothing
    else
        mat_tmp = _vcat(J, DᵀD)
        @bb mat_tmp .*= T(0)
        rhs_tmp = vcat(_vec(fu), _vec(u))
        @bb rhs_tmp .*= T(0)
        linsolve = linsolve_caches(mat_tmp, rhs_tmp, u, p, alg; linsolve_kwargs)
    end

    @bb u_cache = copy(u)
    @bb u_cache_2 = similar(u)
    @bb fu_cache_2 = similar(fu)
    @bb du_cache = similar(du)
    Jv = J * v
    @bb v_cache = similar(v)

    return LevenbergMarquardtCache{iip, fastls}(f, alg, u, u_cache, u_cache_2, fu, fu_cache,
        fu_cache_2, du, du_cache, J, JᵀJ, Jv, DᵀD, v, v_cache, a, mat_tmp, rhs_tmp, p, uf,
        linsolve, jac_cache, false, maxiters, internalnorm, ReturnCode.Default, abstol,
        reltol, prob, λ, λ_factor, damping_increase_factor, damping_decrease_factor, h,
        α_geodesic, b_uphill, min_damping_D, internalnorm(v_cache), loss, make_new_J,
        NLStats(1, 0, 0, 0, 0), tc_cache_1, tc_cache_2, trace)
end

function perform_step!(cache::LevenbergMarquardtCache{iip, fastls}) where {iip, fastls}
    @unpack alg, linsolve = cache

    if cache.make_new_J
        cache.J = jacobian!!(cache.J, cache)
        if fastls
            cache.JᵀJ = __sum_JᵀJ!!(cache.JᵀJ, cache.J)
        else
            @bb cache.JᵀJ = transpose(cache.J) × cache.J
        end
        cache.DᵀD = __update_LM_diagonal!!(cache.DᵀD, cache.JᵀJ)
        cache.make_new_J = false
    end

    # Usual Levenberg-Marquardt step ("velocity").
    # The following lines do: cache.v = -cache.mat_tmp \ cache.u_tmp
    if fastls
        if setindex_trait(cache.mat_tmp) === CanSetindex()
            copyto!(@view(cache.mat_tmp[1:length(cache.fu), :]), cache.J)
            cache.mat_tmp[(length(cache.fu) + 1):end, :] .= cache.λ .* cache.DᵀD
        else
            cache.mat_tmp = _vcat(cache.J, cache.λ .* cache.DᵀD)
        end
        if setindex_trait(cache.rhs_tmp) === CanSetindex()
            cache.rhs_tmp[1:length(cache.fu)] .= _vec(cache.fu)
        else
            cache.rhs_tmp = _vcat(_vec(cache.fu), zero(_vec(cache.u)))
        end
        linres = dolinsolve(alg.precs, linsolve; A = cache.mat_tmp,
            b = cache.rhs_tmp, linu = _vec(cache.v), cache.p, reltol = cache.abstol)
        @bb @. cache.v = -linres.u
    else
        @bb cache.u_cache_2 = transpose(cache.J) × cache.fu
        @bb @. cache.mat_tmp = cache.JᵀJ + cache.λ * cache.DᵀD
        linres = dolinsolve(alg.precs, linsolve; A = cache.mat_tmp,
            b = _vec(cache.u_cache_2), linu = _vec(cache.v), cache.p, reltol = cache.abstol)
        cache.linsolve = linres.cache
        @bb @. cache.v = -linres.u
    end

    update_trace!(cache.trace, cache.stats.nsteps + 1, get_u(cache), get_fu(cache), cache.J,
        cache.v)

    # Geodesic acceleration (step_size = v + a / 2).
    @bb @. cache.u_cache_2 = cache.u + cache.h * cache.v
    evaluate_f(cache, cache.u_cache_2, cache.p, Val(:fu_cache_2))

    # The following lines do: cache.a = -cache.mat_tmp \ cache.fu_tmp
    # NOTE: Don't pass `A` in again, since we want to reuse the previous solve
    @bb cache.Jv = cache.J × cache.v
    @bb @. cache.fu_cache_2 = (2 / cache.h) *
                              ((cache.fu_cache_2 - cache.fu) / cache.h - cache.Jv)
    if fastls
        if setindex_trait(cache.rhs_tmp) === CanSetindex()
            cache.rhs_tmp[1:length(cache.fu)] .= _vec(cache.fu_cache_2)
        else
            cache.rhs_tmp = _vcat(_vec(cache.fu_cache_2), zero(_vec(cache.u)))
        end
        linres = dolinsolve(alg.precs, linsolve; b = cache.rhs_tmp, linu = _vec(cache.a),
            cache.p, reltol = cache.abstol)
        @bb @. cache.a = -linres.u
    else
        @bb cache.u_cache_2 = transpose(J) × cache.fu_cache_2
        linres = dolinsolve(alg.precs, linsolve; b = _vec(cache.u_cache_2),
            linu = _vec(cache.a), cache.p, reltol = cache.abstol)
        cache.linsolve = linres.cache
        @bb @. cache.a = -linres.du
    end

    cache.stats.nsolve += 2
    cache.stats.nfactors += 2

    # Require acceptable steps to satisfy the following condition.
    norm_v = cache.internalnorm(cache.v)
    if 2 * cache.internalnorm(cache.a) ≤ cache.α_geodesic * norm_v
        @bb @. cache.du_cache = cache.v + cache.a / 2
        @bb @. cache.u_cache_2 = cache.u + cache.du_cache
        evaluate_f(cache, cache.u_cache_2, cache.p, Val(:fu_cache_2))
        loss = cache.internalnorm(cache.fu_cache_2)

        # Condition to accept uphill steps (evaluates to `loss ≤ loss_old` in iteration 1).
        β = dot(cache.v, cache.v_cache) / (norm_v * cache.norm_v_old)
        if (1 - β)^cache.b_uphill * loss ≤ cache.loss_old
            # Accept step.
            @bb copyto!(cache.u, cache.u_cache_2)
            check_and_update!(cache.tc_cache_1, cache, cache.fu_cache, cache.u,
                cache.u_cache)
            if !cache.force_stop && cache.tc_cache_2 !== nothing # For NLLS Problems
                @bb @. cache.fu = cache.fu_cache_2 - cache.fu
                check_and_update!(cache.tc_cache_2, cache, cache.fu, cache.u, cache.u_cache)
            end
            @bb copyto!(cache.fu, cache.fu_cache_2)
            @bb copyto!(cache.v_cache, cache.v)
            cache.norm_v_old = norm_v
            cache.loss_old = loss
            cache.λ_factor = 1 / cache.damping_decrease_factor
            cache.make_new_J = true
        end
    end

    @bb copyto!(cache.u_cache, cache.u)
    cache.λ *= cache.λ_factor
    cache.λ_factor = cache.damping_increase_factor
    return nothing
end

function __reinit_internal!(cache::LevenbergMarquardtCache;
        termination_condition = get_termination_mode(cache.tc_cache_1), kwargs...)
    abstol, reltol, tc_cache_1 = init_termination_cache(cache.abstol, cache.reltol,
        cache.fu, cache.u, termination_condition)
    if cache.tc_cache_2 !== nothing
        _, _, tc_cache_2 = init_termination_cache(cache.abstol, cache.reltol, cache.fu,
            cache.u, termination_condition)
        cache.tc_cache_2 = tc_cache_2
    end

    cache.tc_cache_1 = tc_cache_1
    cache.abstol = abstol
    cache.reltol = reltol
    return nothing
end
