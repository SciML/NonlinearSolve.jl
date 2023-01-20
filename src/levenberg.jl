"""
```julia
LevenbergMarquardt(; chunk_size = Val{0}(), autodiff = Val{true}(),
                standardtag = Val{true}(), concrete_jac = nothing,
                diff_type = Val{:forward}, linsolve = nothing, precs = DEFAULT_PRECS)
```

An advanced LevenbergMarquardt implementation with support for efficient handling of sparse
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
    b_uphill::T  # TODO test what happens if the input to this in an Interger, but Float to the others
    min_damping_D::T
end

function LevenbergMarquardt(; chunk_size = Val{0}(), autodiff = Val{true}(),
                            standardtag = Val{true}(), concrete_jac = nothing,
                            diff_type = Val{:forward}, linsolve = nothing,
                            precs = DEFAULT_PRECS,
                            damping_initial::Real = 1.0,
                            damping_increase_factor::Real = 2.0,
                            damping_decrease_factor::Real = 3.0,
                            finite_diff_step_geodesic::Real = 0.1,
                            α_geodesic::Real = 0.75,
                            b_uphill::Real = 1.0,
                            min_damping_D::Real = 1e-6)
    LevenbergMarquardt{_unwrap_val(chunk_size), _unwrap_val(autodiff), diff_type,
                       typeof(linsolve), typeof(precs), _unwrap_val(standardtag),
                       _unwrap_val(concrete_jac), typeof(damping_initial)}(linsolve, precs,
                                                                           damping_initial,
                                                                           damping_increase_factor,
                                                                           damping_decrease_factor,
                                                                           finite_diff_step_geodesic,
                                                                           α_geodesic,
                                                                           b_uphill,
                                                                           min_damping_D)
end

mutable struct LevenbergMarquardtCache{iip, fType, algType, uType, duType, resType, pType,
                                       INType, tolType,
                                       probType, ufType, L, jType, JC, DᵀDType,
                                       floatType
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
    DᵀD::DᵀDType
    JᵀJ::jType
    λ::floatType
    λ_factor::floatType         # TODO test if this handles interger inputs. (test to input damp_init::float and damp_inc::int)
    v::uType
    a::uType
    tmp_vec::uType
    v_old::uType
    norm_v_old::floatType
    δ::uType
    loss_old::floatType

    function LevenbergMarquardtCache{iip}(f::fType, alg::algType, u::uType, fu::resType,
                                          p::pType,
                                          uf::ufType, linsolve::L, J::jType, du1::duType,
                                          jac_config::JC, iter::Int,
                                          force_stop::Bool, maxiters::Int,
                                          internalnorm::INType,
                                          retcode::SciMLBase.ReturnCode.T, abstol::tolType,
                                          prob::probType, DᵀD::DᵀDType, JᵀJ::jType,
                                          λ::floatType, λ_factor::floatType, v::uType,
                                          a::uType, tmp_vec::uType,
                                          v_old::uType, norm_v_old::floatType, δ::uType,
                                          loss_old::floatType) where {
                                                                      iip, fType, algType,
                                                                      uType,
                                                                      duType, resType,
                                                                      pType,
                                                                      INType,
                                                                      tolType,
                                                                      probType, ufType, L,
                                                                      jType, JC,
                                                                      DᵀDType,
                                                                      floatType
                                                                      }
        new{iip, fType, algType, uType, duType, resType,
            pType, INType, tolType, probType, ufType, L,
            jType, JC, DᵀDType, floatType}(f,
                                                                   alg,
                                                                   u,
                                                                   fu,
                                                                   p,
                                                                   uf,
                                                                   linsolve,
                                                                   J,
                                                                   du1,
                                                                   jac_config,
                                                                   iter,
                                                                   force_stop,
                                                                   maxiters,
                                                                   internalnorm,
                                                                   retcode,
                                                                   abstol,
                                                                   prob,
                                                                   DᵀD,
                                                                   JᵀJ,
                                                                   λ,
                                                                   λ_factor,
                                                                   v,
                                                                   a,
                                                                   tmp_vec,
                                                                   v_old,
                                                                   norm_v_old,
                                                                   δ,
                                                                   loss_old)
    end
end

function jacobian_caches(alg::LevenbergMarquardt, f, u, p, ::Val{true})
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

function jacobian_caches(alg::LevenbergMarquardt, f, u, p, ::Val{false})
    JacobianWrapper(f, p), nothing, ArrayInterfaceCore.undefmatrix(u), nothing, nothing
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
    uf, linsolve, J, du1, jac_config = jacobian_caches(alg, f, u, p, Val(iip))

    # "A good compromise is to specify a minimum value of the damping terms in DᵀD".
    if u isa Number
        DᵀD = alg.min_damping_D # TODO check if type is correct.
    else
        d = d = similar(u)
        d .= alg.min_damping_D
        DᵀD = Diagonal(d)
    end
    loss = internalnorm(fu)
    JᵀJ = zero(J)
    v = zero(u)
    a = zero(u)
    tmp_vec = zero(u)
    v_old = zero(u)
    δ = zero(u)

    return LevenbergMarquardtCache{iip}(f, alg, u, fu, p, uf, linsolve, J, du1, jac_config,
                                        1, false, maxiters, internalnorm,
                                        ReturnCode.Default, abstol, prob, DᵀD, JᵀJ,
                                        alg.damping_initial, alg.damping_increase_factor,
                                        v, a, tmp_vec, v_old, loss, δ, loss)
end

function perform_step!(cache::LevenbergMarquardtCache{true})
    # TODO implement for in-place...
    # @unpack fu, f = cache
    # if iszero(fu)
    #     cache.force_stop = true
    #     return nothing
    # end

    # cache.J = jacobian(cache, f)
    # @unpack u, f, p, λ, J = cache
    # mul!(cache.JᵀJ, J', J)
    # cache.DᵀD .= max.(cache.DᵀD, Diagonal(cache.JᵀJ))

    # # Usual Levenberg-Marquardt step ("velocity").
    # cache.v = -(cache.JᵀJ + λ * cache.DᵀD) \ (J' * fu)

    # @unpack v, alg = cache
    # h = alg.finite_diff_step_geodesic
    # # Geodesic acceleration (step_size = v + a / 2).
    # cache.a = -J \ ((2 / h) .* ((f(u .+ h * v, p) .- fu) ./ h .- mul!(cache.Jv, J, v)))

    # # Require acceptable steps to satisfy the following condition.
    # norm_v = norm(v)
    # if (2 * norm(cache.a) / norm_v) < alg.α_geodesic
    #     cache.δ = v + cache.a / 2
    #     @unpack δ, loss_old, norm_v_old, v_old = cache
    #     fu_new = f(u .+ δ, p)
    #     loss = cache.internalnorm(fu_new)

    #     # Condition to accept uphill steps (evaluates to `loss ≤ loss_old` in iteration 1).
    #     β = dot(v, v_old) / (norm_v * norm_v_old)
    #     if (1 - β)^alg.b_uphill * loss ≤ loss_old
    #         # Accept step.
    #         cache.u += δ
    #         if loss < cache.abstol
    #             cache.force_stop = true
    #             return nothing
    #         end
    #         cache.fu = fu_new
    #         cache.v_old = v
    #         cache.norm_v_old = norm_v
    #         cache.loss_old = loss
    #         cache.λ_factor = 1 / alg.damping_decrease_factor
    #     end
    # end
    # cache.λ *= cache.λ_factor
    # cache.λ_factor = alg.damping_increase_factor
    return nothing
end

function perform_step!(cache::LevenbergMarquardtCache{false})
    @unpack fu, f = cache
    if iszero(fu)
        cache.force_stop = true
        return nothing
    end

    J = jacobian(cache, f)
    @unpack u, f, p, λ = cache
    cache.JᵀJ = J' * J
    if cache.DᵀD isa Number
        cache.DᵀD = max(cache.DᵀD, cache.JᵀJ)
    else
        cache.DᵀD .= max.(cache.DᵀD, Diagonal(cache.JᵀJ))
    end

    # Usual Levenberg-Marquardt step ("velocity").
    cache.v = -(cache.JᵀJ + λ * cache.DᵀD) \ (J' * fu)

    @unpack v, alg = cache
    h = alg.finite_diff_step_geodesic
    # Geodesic acceleration (step_size = v + a / 2).
    cache.a = -J \ ((2 / h) .* ((f(u .+ h * v, p) .- fu) ./ h .- J * v))

    # Require acceptable steps to satisfy the following condition.
    norm_v = norm(v)
    if (2 * norm(cache.a) / norm_v) < alg.α_geodesic
        cache.δ = v + cache.a / 2
        @unpack δ, loss_old, norm_v_old, v_old = cache
        fu_new = f(u .+ δ, p)
        loss = cache.internalnorm(fu_new)

        # Condition to accept uphill steps (evaluates to `loss ≤ loss_old` in iteration 1).
        β = dot(v, v_old) / (norm_v * norm_v_old)
        if (1 - β)^alg.b_uphill * loss ≤ loss_old
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
            cache.λ_factor = 1 / alg.damping_decrease_factor
        end
    end
    cache.λ *= cache.λ_factor
    cache.λ_factor = alg.damping_increase_factor
    return nothing
end

function SciMLBase.solve!(cache::LevenbergMarquardtCache)
    while !cache.force_stop && cache.iter < cache.maxiters
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
