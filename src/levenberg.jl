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
@concrete struct LevenbergMarquardt{CJ, AD, T} <: AbstractNewtonAlgorithm{CJ, AD}
    ad::AD
    linsolve
    precs
    damping_initial::T
    damping_increase_factor::T
    damping_decrease_factor::T
    finite_diff_step_geodesic::T
    α_geodesic::T
    b_uphill::T
    min_damping_D::T
end

function set_ad(alg::LevenbergMarquardt{CJ}, ad) where {CJ}
    return LevenbergMarquardt{CJ}(ad, alg.linsolve, alg.precs, alg.damping_initial,
        alg.damping_increase_factor, alg.damping_decrease_factor,
        alg.finite_diff_step_geodesic, alg.α_geodesic, alg.b_uphill, alg.min_damping_D)
end

function LevenbergMarquardt(; concrete_jac = nothing, linsolve = nothing,
    precs = DEFAULT_PRECS, damping_initial::Real = 1.0, damping_increase_factor::Real = 2.0,
    damping_decrease_factor::Real = 3.0, finite_diff_step_geodesic::Real = 0.1,
    α_geodesic::Real = 0.75, b_uphill::Real = 1.0, min_damping_D::AbstractFloat = 1e-8,
    adkwargs...)
    ad = default_adargs_to_adtype(; adkwargs...)
    return LevenbergMarquardt{_unwrap_val(concrete_jac)}(ad, linsolve, precs,
        damping_initial, damping_increase_factor, damping_decrease_factor,
        finite_diff_step_geodesic, α_geodesic, b_uphill, min_damping_D)
end

@concrete mutable struct LevenbergMarquardtCache{iip, fastqr} <:
                         AbstractNonlinearSolveCache{iip}
    f
    alg
    u
    fu1
    fu2
    du
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
    DᵀD
    JᵀJ
    λ
    λ_factor
    damping_increase_factor
    damping_decrease_factor
    h
    α_geodesic
    b_uphill
    min_damping_D
    v
    a
    tmp_vec
    v_old
    norm_v_old
    δ
    loss_old
    make_new_J::Bool
    fu_tmp
    u_tmp
    Jv
    mat_tmp
    rhs_tmp
    J²
    stats::NLStats
end

function SciMLBase.__init(prob::Union{NonlinearProblem{uType, iip},
        NonlinearLeastSquaresProblem{uType, iip}}, alg_::LevenbergMarquardt,
    args...; alias_u0 = false, maxiters = 1000, abstol = 1e-6, internalnorm = DEFAULT_NORM,
    linsolve_kwargs = (;), kwargs...) where {uType, iip}
    alg = get_concrete_algorithm(alg_, prob)
    @unpack f, u0, p = prob
    u = alias_u0 ? u0 : deepcopy(u0)
    fu1 = evaluate_f(prob, u)

    if !needs_square_A(alg.linsolve) && !(u isa Number) && !(u isa StaticArray)
        linsolve_with_JᵀJ = Val(false)
    else
        linsolve_with_JᵀJ = Val(true)
    end

    if _unwrap_val(linsolve_with_JᵀJ)
        uf, linsolve, J, fu2, jac_cache, du, JᵀJ, v = jacobian_caches(alg, f, u, p,
            Val(iip); linsolve_kwargs, linsolve_with_JᵀJ)
        J² = nothing
    else
        uf, linsolve, J, fu2, jac_cache, du = jacobian_caches(alg, f, u, p, Val(iip);
            linsolve_kwargs, linsolve_with_JᵀJ)
        JᵀJ = similar(u)
        J² = similar(J)
        v = similar(du)
    end

    λ = convert(eltype(u), alg.damping_initial)
    λ_factor = convert(eltype(u), alg.damping_increase_factor)
    damping_increase_factor = convert(eltype(u), alg.damping_increase_factor)
    damping_decrease_factor = convert(eltype(u), alg.damping_decrease_factor)
    h = convert(eltype(u), alg.finite_diff_step_geodesic)
    α_geodesic = convert(eltype(u), alg.α_geodesic)
    b_uphill = convert(eltype(u), alg.b_uphill)
    min_damping_D = convert(eltype(u), alg.min_damping_D)

    if u isa Number
        DᵀD = min_damping_D
    else
        d = similar(u)
        d .= min_damping_D
        DᵀD = Diagonal(_vec(d))
    end

    loss = internalnorm(fu1)
    a = _mutable_zero(u)
    tmp_vec = _mutable_zero(u)
    v_old = _mutable_zero(u)
    δ = _mutable_zero(u)
    make_new_J = true
    fu_tmp = zero(fu1)

    if _unwrap_val(linsolve_with_JᵀJ)
        mat_tmp = zero(JᵀJ)
        rhs_tmp = nothing
    else
        # Preserve Types
        mat_tmp = vcat(J, DᵀD)
        fill!(mat_tmp, zero(eltype(u)))
        rhs_tmp = vcat(fu1, u)
        fill!(rhs_tmp, zero(eltype(u)))
        linsolve = __setup_linsolve(mat_tmp, rhs_tmp, u, p, alg)
    end

    return LevenbergMarquardtCache{iip, !_unwrap_val(linsolve_with_JᵀJ)}(f, alg, u, fu1,
        fu2, du, p, uf, linsolve, J,
        jac_cache, false, maxiters, internalnorm, ReturnCode.Default, abstol, prob, DᵀD,
        JᵀJ, λ, λ_factor, damping_increase_factor, damping_decrease_factor, h, α_geodesic,
        b_uphill, min_damping_D, v, a, tmp_vec, v_old, loss, δ, loss, make_new_J, fu_tmp,
        zero(u), zero(fu1), mat_tmp, rhs_tmp, J², NLStats(1, 0, 0, 0, 0))
end

function perform_step!(cache::LevenbergMarquardtCache{true, fastqr}) where {fastqr}
    @unpack fu1, f, make_new_J = cache
    if iszero(fu1)
        cache.force_stop = true
        return nothing
    end

    if make_new_J
        jacobian!!(cache.J, cache)
        if fastqr
            cache.J² .= cache.J .^ 2
            sum!(cache.JᵀJ', cache.J²)
            cache.DᵀD.diag .= max.(cache.DᵀD.diag, cache.JᵀJ)
        else
            __matmul!(cache.JᵀJ, cache.J', cache.J)
            cache.DᵀD .= max.(cache.DᵀD, Diagonal(cache.JᵀJ))
        end
        cache.make_new_J = false
        cache.stats.njacs += 1
    end
    @unpack u, p, λ, JᵀJ, DᵀD, J, alg, linsolve = cache

    # Usual Levenberg-Marquardt step ("velocity").
    # The following lines do: cache.v = -cache.mat_tmp \ cache.u_tmp
    if fastqr
        cache.mat_tmp[1:length(fu1), :] .= cache.J
        cache.mat_tmp[(length(fu1) + 1):end, :] .= λ .* cache.DᵀD
        cache.rhs_tmp[1:length(fu1)] .= _vec(fu1)
        linres = dolinsolve(alg.precs, linsolve; A = cache.mat_tmp,
            b = cache.rhs_tmp, linu = _vec(cache.du), p = p, reltol = cache.abstol)
        _vec(cache.v) .= -_vec(cache.du)
    else
        mul!(_vec(cache.u_tmp), J', _vec(fu1))
        @. cache.mat_tmp = JᵀJ + λ * DᵀD
        linres = dolinsolve(alg.precs, linsolve; A = __maybe_symmetric(cache.mat_tmp),
            b = _vec(cache.u_tmp), linu = _vec(cache.du), p = p, reltol = cache.abstol)
        cache.linsolve = linres.cache
        _vec(cache.v) .= -_vec(cache.du)
    end

    # Geodesic acceleration (step_size = v + a / 2).
    @unpack v, α_geodesic, h = cache
    cache.u_tmp .= _restructure(cache.u_tmp, _vec(u) .+ h .* _vec(v))
    f(cache.fu_tmp, cache.u_tmp, p)

    # The following lines do: cache.a = -J \ cache.fu_tmp
    # NOTE: Don't pass `A` in again, since we want to reuse the previous solve
    mul!(_vec(cache.Jv), J, _vec(v))
    @. cache.fu_tmp = (2 / h) * ((cache.fu_tmp - fu1) / h - cache.Jv)
    if fastqr
        cache.rhs_tmp[1:length(fu1)] .= _vec(cache.fu_tmp)
        linres = dolinsolve(alg.precs, linsolve; b = cache.rhs_tmp, linu = _vec(cache.du),
            p = p, reltol = cache.abstol)
    else
        mul!(_vec(cache.u_tmp), J', _vec(cache.fu_tmp))
        linres = dolinsolve(alg.precs, linsolve; b = _vec(cache.u_tmp),
            linu = _vec(cache.du), p = p, reltol = cache.abstol)
        cache.linsolve = linres.cache
        @. cache.a = -cache.du
    end
    cache.stats.nsolve += 2
    cache.stats.nfactors += 2

    # Require acceptable steps to satisfy the following condition.
    norm_v = norm(v)
    if 2 * norm(cache.a) ≤ α_geodesic * norm_v
        _vec(cache.δ) .= _vec(v) .+ _vec(cache.a) ./ 2
        @unpack δ, loss_old, norm_v_old, v_old, b_uphill = cache
        f(cache.fu_tmp, u .+ δ, p)
        cache.stats.nf += 1
        loss = cache.internalnorm(cache.fu_tmp)

        # Condition to accept uphill steps (evaluates to `loss ≤ loss_old` in iteration 1).
        β = dot(v, v_old) / (norm_v * norm_v_old)
        if (1 - β)^b_uphill * loss ≤ loss_old
            # Accept step.
            cache.u .+= δ
            if loss < cache.abstol
                cache.force_stop = true
                return nothing
            end
            cache.fu1 .= cache.fu_tmp
            _vec(cache.v_old) .= _vec(v)
            cache.norm_v_old = norm_v
            cache.loss_old = loss
            cache.λ_factor = 1 / cache.damping_decrease_factor
            cache.make_new_J = true
        end
    end
    cache.λ *= cache.λ_factor
    cache.λ_factor = cache.damping_increase_factor
    return nothing
end

function perform_step!(cache::LevenbergMarquardtCache{false, fastqr}) where {fastqr}
    @unpack fu1, f, make_new_J = cache
    if iszero(fu1)
        cache.force_stop = true
        return nothing
    end

    if make_new_J
        cache.J = jacobian!!(cache.J, cache)
        if fastqr
            cache.JᵀJ = _vec(sum(cache.J .^ 2; dims = 1))
            cache.DᵀD.diag .= max.(cache.DᵀD.diag, cache.JᵀJ)
        else
            cache.JᵀJ = cache.J' * cache.J
            if cache.JᵀJ isa Number
                cache.DᵀD = max(cache.DᵀD, cache.JᵀJ)
            else
                cache.DᵀD .= max.(cache.DᵀD, Diagonal(cache.JᵀJ))
            end
        end
        cache.make_new_J = false
        cache.stats.njacs += 1
    end
    @unpack u, p, λ, JᵀJ, DᵀD, J, linsolve, alg = cache

    # Usual Levenberg-Marquardt step ("velocity").
    if fastqr
        cache.mat_tmp = vcat(J, λ .* cache.DᵀD)
        cache.rhs_tmp[1:length(fu1)] .= -_vec(fu1)
        linres = dolinsolve(alg.precs, linsolve; A = cache.mat_tmp,
            b = cache.rhs_tmp, linu = _vec(cache.v), p = p, reltol = cache.abstol)
    else
        cache.mat_tmp = JᵀJ + λ * DᵀD
        if linsolve === nothing
            cache.v = -cache.mat_tmp \ (J' * fu1)
        else
            linres = dolinsolve(alg.precs, linsolve; A = -__maybe_symmetric(cache.mat_tmp),
                b = _vec(J' * _vec(fu1)), linu = _vec(cache.v), p, reltol = cache.abstol)
            cache.linsolve = linres.cache
        end
    end

    @unpack v, h, α_geodesic = cache
    # Geodesic acceleration (step_size = v + a / 2).
    rhs_term = _vec(((2 / h) .* ((_vec(f(u .+ h .* _restructure(u, v), p)) .-
                       _vec(fu1)) ./ h .- J * _vec(v))))
    if fastqr
        cache.rhs_tmp[1:length(fu1)] .= -_vec(rhs_term)
        linres = dolinsolve(alg.precs, linsolve;
            b = cache.rhs_tmp, linu = _vec(cache.a), p = p, reltol = cache.abstol)
    else
        if linsolve === nothing
            cache.a = -cache.mat_tmp \ _vec(J' * rhs_term)
        else
            linres = dolinsolve(alg.precs, linsolve; b = _mutable(_vec(J' * rhs_term)),
                linu = _vec(cache.a), p, reltol = cache.abstol)
            cache.linsolve = linres.cache
        end
    end
    cache.stats.nsolve += 1
    cache.stats.nfactors += 1

    # Require acceptable steps to satisfy the following condition.
    norm_v = norm(v)
    if 2 * norm(cache.a) ≤ α_geodesic * norm_v
        cache.δ = _restructure(cache.δ, _vec(v) .+ _vec(cache.a) ./ 2)
        @unpack δ, loss_old, norm_v_old, v_old, b_uphill = cache
        fu_new = f(u .+ δ, p)
        cache.stats.nf += 1
        loss = cache.internalnorm(fu_new)

        # Condition to accept uphill steps (evaluates to `loss ≤ loss_old` in iteration 1).
        β = dot(v, v_old) / (norm_v * norm_v_old)
        if (1 - β)^b_uphill * loss ≤ loss_old
            # Accept step.
            cache.u += δ
            if loss < cache.abstol
                cache.force_stop = true
                return nothing
            end
            cache.fu1 = fu_new
            cache.v_old = _restructure(cache.v_old, v)
            cache.norm_v_old = norm_v
            cache.loss_old = loss
            cache.λ_factor = 1 / cache.damping_decrease_factor
            cache.make_new_J = true
        end
    end
    cache.λ *= cache.λ_factor
    cache.λ_factor = cache.damping_increase_factor
    return nothing
end
