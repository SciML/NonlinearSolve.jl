"""
```julia
LevenbergMarquardt(; chunk_size = Val{0}(),
                    autodiff = Val{true}(),
                    standardtag = Val{true}(),
                    concrete_jac = nothing,
                    diff_type = Val{:forward},
                    linsolve = nothing, precs = DEFAULT_PRECS,
                    damping_initial::Real = 1.0,
                    damping_increase_factor::Real = 2.0,
                    damping_decrease_factor::Real = 3.0,
                    finite_diff_step_geodesic::Real = 0.1,
                    α_geodesic::Real = 0.75,
                    b_uphill::Real = 1.0,
                    min_damping_D::AbstractFloat = 1e-8)
```

An advanced Levenberg-Marquardt implementation with the improvements suggested in the
[paper](https://arxiv.org/abs/1201.5885) "Improvements to the Levenberg-Marquardt
algorithm for nonlinear least-squares minimization". Designed for large-scale and
numerically-difficult nonlinear systems.


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


!!! note

    Currently, the linear solver and chunk size choice only applies to in-place defined
    `NonlinearProblem`s. That is expected to change in the future.
"""
struct LevenbergMarquardt{CS, AD, FDT, L, P, ST, CJ, T} <:
       AbstractNewtonAlgorithm{CS, AD, FDT, ST, CJ}
    linsolve::L
    precs::P
    damping_initial::T
    damping_increase_factor::T
    damping_decrease_factor::T
    finite_diff_step_geodesic::T
    α_geodesic::T
    b_uphill::T
    min_damping_D::T
end

function LevenbergMarquardt(; chunk_size = Val{0}(),
    autodiff = Val{true}(),
    standardtag = Val{true}(),
    concrete_jac = nothing,
    diff_type = Val{:forward},
    linsolve = nothing,
    precs = DEFAULT_PRECS,
    damping_initial::Real = 1.0,
    damping_increase_factor::Real = 2.0,
    damping_decrease_factor::Real = 3.0,
    finite_diff_step_geodesic::Real = 0.1,
    α_geodesic::Real = 0.75,
    b_uphill::Real = 1.0,
    min_damping_D::AbstractFloat = 1e-8)
    LevenbergMarquardt{_unwrap_val(chunk_size), _unwrap_val(autodiff), diff_type,
        typeof(linsolve), typeof(precs), _unwrap_val(standardtag),
        _unwrap_val(concrete_jac),
        typeof(min_damping_D)}(linsolve, precs,
        damping_initial,
        damping_increase_factor,
        damping_decrease_factor,
        finite_diff_step_geodesic,
        α_geodesic,
        b_uphill,
        min_damping_D)
end

mutable struct LevenbergMarquardtCache{iip, fType, algType, uType, duType, resType, pType,
    INType, tolType, probType, ufType, L, jType, JC,
    DᵀDType, λType, lossType,
}
    f::fType
    alg::algType
    u::uType
    fu::resType
    p::pType
    uf::ufType
    linsolve::L
    J::jType
    du_tmp::duType
    jac_config::JC
    force_stop::Bool
    maxiters::Int
    internalnorm::INType
    retcode::SciMLBase.ReturnCode.T
    abstol::tolType
    prob::probType
    DᵀD::DᵀDType
    JᵀJ::jType
    λ::λType
    λ_factor::λType
    damping_increase_factor::λType
    damping_decrease_factor::λType
    h::λType
    α_geodesic::λType
    b_uphill::λType
    min_damping_D::λType
    v::uType
    a::uType
    tmp_vec::uType
    v_old::uType
    norm_v_old::lossType
    δ::uType
    loss_old::lossType
    make_new_J::Bool
    fu_tmp::resType
    mat_tmp::jType
    stats::NLStats

    function LevenbergMarquardtCache{iip}(f::fType, alg::algType, u::uType, fu::resType,
        p::pType, uf::ufType, linsolve::L, J::jType,
        du_tmp::duType, jac_config::JC,
        force_stop::Bool, maxiters::Int,
        internalnorm::INType,
        retcode::SciMLBase.ReturnCode.T, abstol::tolType,
        prob::probType, DᵀD::DᵀDType, JᵀJ::jType,
        λ::λType, λ_factor::λType,
        damping_increase_factor::λType,
        damping_decrease_factor::λType, h::λType,
        α_geodesic::λType, b_uphill::λType,
        min_damping_D::λType, v::uType,
        a::uType, tmp_vec::uType, v_old::uType,
        norm_v_old::lossType, δ::uType,
        loss_old::lossType, make_new_J::Bool,
        fu_tmp::resType,
        mat_tmp::jType,
        stats::NLStats) where {
        iip, fType, algType,
        uType, duType, resType,
        pType, INType, tolType,
        probType, ufType, L,
        jType, JC, DᵀDType,
        λType, lossType,
    }
        new{iip, fType, algType, uType, duType, resType,
            pType, INType, tolType, probType, ufType, L,
            jType, JC, DᵀDType, λType, lossType}(f, alg, u, fu, p, uf, linsolve, J, du_tmp,
            jac_config, force_stop, maxiters,
            internalnorm, retcode, abstol, prob, DᵀD,
            JᵀJ, λ, λ_factor,
            damping_increase_factor,
            damping_decrease_factor, h,
            α_geodesic, b_uphill, min_damping_D,
            v, a, tmp_vec, v_old,
            norm_v_old, δ, loss_old, make_new_J,
            fu_tmp, mat_tmp, stats)
    end
end

function jacobian_caches(alg::LevenbergMarquardt, f, u, p, ::Val{true})
    uf = JacobianWrapper(f, p)
    J = ArrayInterface.undefmatrix(u)

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

function jacobian_caches(alg::LevenbergMarquardt, f, u, p, ::Val{false})
    JacobianWrapper(f, p), nothing, ArrayInterface.undefmatrix(u), nothing, nothing
end

function SciMLBase.__init(prob::NonlinearProblem{uType, iip}, alg::LevenbergMarquardt,
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
    uf, linsolve, J, du_tmp, jac_config = jacobian_caches(alg, f, u, p, Val(iip))

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
        DᵀD = Diagonal(d)
    end

    loss = internalnorm(fu)
    JᵀJ = zero(J)
    v = zero(u)
    a = zero(u)
    tmp_vec = zero(u)
    v_old = zero(u)
    δ = zero(u)
    make_new_J = true
    fu_tmp = zero(fu)
    mat_tmp = zero(J)

    return LevenbergMarquardtCache{iip}(f, alg, u, fu, p, uf, linsolve, J, du_tmp,
        jac_config, false, maxiters, internalnorm,
        ReturnCode.Default, abstol, prob, DᵀD, JᵀJ,
        λ, λ_factor, damping_increase_factor,
        damping_decrease_factor, h,
        α_geodesic, b_uphill, min_damping_D,
        v, a, tmp_vec, v_old, loss, δ, loss, make_new_J,
        fu_tmp, mat_tmp, NLStats(1, 0, 0, 0, 0))
end
function perform_step!(cache::LevenbergMarquardtCache{true})
    @unpack fu, f, make_new_J = cache
    if iszero(fu)
        cache.force_stop = true
        return nothing
    end
    if make_new_J
        jacobian!(cache.J, cache)
        mul!(cache.JᵀJ, cache.J', cache.J)
        cache.DᵀD .= max.(cache.DᵀD, Diagonal(cache.JᵀJ))
        cache.make_new_J = false
        cache.stats.njacs += 1
    end
    @unpack u, p, λ, JᵀJ, DᵀD, J, alg, linsolve = cache

    # Usual Levenberg-Marquardt step ("velocity").
    # The following lines do: cache.v = -cache.mat_tmp \ cache.fu_tmp
    mul!(cache.fu_tmp, J', fu)
    @. cache.mat_tmp = JᵀJ + λ * DᵀD
    linres = dolinsolve(alg.precs, linsolve, A = cache.mat_tmp, b = _vec(cache.fu_tmp),
        linu = _vec(cache.du_tmp), p = p, reltol = cache.abstol)
    cache.linsolve = linres.cache
    @. cache.v = -cache.du_tmp

    # Geodesic acceleration (step_size = v + a / 2).
    @unpack v, α_geodesic, h = cache
    f(cache.fu_tmp, u .+ h .* v, p)

    # The following lines do: cache.a = -J \ cache.fu_tmp
    mul!(cache.du_tmp, J, v)
    @. cache.fu_tmp = (2 / h) * ((cache.fu_tmp - fu) / h - cache.du_tmp)
    linres = dolinsolve(alg.precs, linsolve, A = J, b = _vec(cache.fu_tmp),
        linu = _vec(cache.du_tmp), p = p, reltol = cache.abstol)
    cache.linsolve = linres.cache
    @. cache.a = -cache.du_tmp
    cache.stats.nsolve += 2
    cache.stats.nfactors += 2

    # Require acceptable steps to satisfy the following condition.
    norm_v = norm(v)
    if (2 * norm(cache.a) / norm_v) < α_geodesic
        @. cache.δ = v + cache.a / 2
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
            cache.fu .= cache.fu_tmp
            cache.v_old .= v
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

function perform_step!(cache::LevenbergMarquardtCache{false})
    @unpack fu, f, make_new_J = cache
    if iszero(fu)
        cache.force_stop = true
        return nothing
    end
    if make_new_J
        cache.J = jacobian(cache, f)
        cache.JᵀJ = cache.J' * cache.J
        if cache.JᵀJ isa Number
            cache.DᵀD = max(cache.DᵀD, cache.JᵀJ)
        else
            cache.DᵀD .= max.(cache.DᵀD, Diagonal(cache.JᵀJ))
        end
        cache.make_new_J = false
        cache.stats.njacs += 1
    end
    @unpack u, p, λ, JᵀJ, DᵀD, J = cache

    # Usual Levenberg-Marquardt step ("velocity").
    cache.v = -(JᵀJ + λ * DᵀD) \ (J' * fu)

    @unpack v, h, α_geodesic = cache
    # Geodesic acceleration (step_size = v + a / 2).
    cache.a = -J \ ((2 / h) .* ((f(u .+ h .* v, p) .- fu) ./ h .- J * v))
    cache.stats.nsolve += 1
    cache.stats.nfactors += 1

    # Require acceptable steps to satisfy the following condition.
    norm_v = norm(v)
    if (2 * norm(cache.a) / norm_v) < α_geodesic
        cache.δ = v .+ cache.a ./ 2
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
            cache.fu = fu_new
            cache.v_old = v
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

function SciMLBase.solve!(cache::LevenbergMarquardtCache)
    while !cache.force_stop && cache.stats.nsteps < cache.maxiters
        perform_step!(cache)
        cache.stats.nsteps += 1
    end

    if cache.stats.nsteps == cache.maxiters
        cache.retcode = ReturnCode.MaxIters
    else
        cache.retcode = ReturnCode.Success
    end

    SciMLBase.build_solution(cache.prob, cache.alg, cache.u, cache.fu;
        retcode = cache.retcode, stats = cache.stats)
end
