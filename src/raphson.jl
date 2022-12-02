struct NewtonRaphson{CS, AD, FDT, L, P, ST, CJ} <:
       AbstractNewtonAlgorithm{CS, AD, FDT, ST, CJ}
    linsolve::L
    precs::P
end

function NewtonRaphson(; chunk_size = Val{0}(), autodiff = Val{true}(),
                       standardtag = Val{true}(), concrete_jac = nothing,
                       diff_type = Val{:forward}, linsolve = nothing, precs = DEFAULT_PRECS)
    NewtonRaphson{_unwrap_val(chunk_size), _unwrap_val(autodiff), diff_type,
                  typeof(linsolve), typeof(precs), _unwrap_val(standardtag),
                  _unwrap_val(concrete_jac)}(linsolve, precs)
end

mutable struct NewtonRaphsonCache{iip, fType, algType, uType, duType, resType, pType,
                                  INType, tolType,
                                  probType, ufType, L, jType, JC}
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

    function NewtonRaphsonCache{iip}(f::fType, alg::algType, u::uType, fu::resType,
                                     p::pType,
                                     uf::ufType, linsolve::L, J::jType, du1::duType,
                                     jac_config::JC, iter::Int,
                                     force_stop::Bool, maxiters::Int, internalnorm::INType,
                                     retcode::SciMLBase.ReturnCode.T, abstol::tolType,
                                     prob::probType) where {
                                                            iip, fType, algType, uType,
                                                            duType, resType, pType, INType,
                                                            tolType,
                                                            probType, ufType, L, jType, JC}
        new{iip, fType, algType, uType, duType, resType, pType, INType, tolType,
            probType, ufType, L, jType, JC}(f, alg, u, fu, p,
                                            uf, linsolve, J, du1, jac_config, iter,
                                            force_stop, maxiters, internalnorm,
                                            retcode, abstol, prob)
    end
end

function jacobian_caches(alg::NewtonRaphson, f, u, p, ::Val{true})
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
    tmp = zero(u)
    if alg_autodiff(alg)
        jac_config = ForwardDiff.JacobianConfig(uf, du1, u)
    else
        if alg.diff_type != Val{:complex}
            du2 = zero(u)
            jac_config = FiniteDiff.JacobianCache(tmp, du1, du2, alg.diff_type)
        else
            jac_config = FiniteDiff.JacobianCache(Complex{eltype(tmp)}.(tmp),
                                                  Complex{eltype(du1)}.(du1), nothing,
                                                  alg.diff_type, eltype(u))
        end
    end
    uf, linsolve, J, du1, jac_config
end

function jacobian_caches(alg::NewtonRaphson, f, u, p, ::Val{false})
    nothing, nothing, nothing, nothing, nothing
end

function SciMLBase.__init(prob::NonlinearProblem{uType, iip}, alg::NewtonRaphson,
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

    return NewtonRaphsonCache{iip}(f, alg, u, fu, p, uf, linsolve, J, du1, jac_config,
                                   1, false, maxiters, internalnorm,
                                   ReturnCode.Default, abstol, prob)
end

function perform_step!(cache::NewtonRaphsonCache{true})
    @unpack u, fu, f, p, alg = cache
    @unpack J, linsolve, du1 = cache
    calc_J!(J, cache, cache)

    # u = u - J \ fu
    linres = dolinsolve(alg.precs, linsolve, A = J, b = fu, linu = du1,
                        p = p, reltol = cache.abstol)
    cache.linsolve = linres.cache
    @. u = u - du1
    f(fu, u, p)

    if cache.internalnorm(cache.fu) < cache.abstol
        cache.force_stop = true
    end
    return nothing
end

function perform_step!(cache::NewtonRaphsonCache{false})
    @unpack u, fu, f, p = cache
    J = calc_J(cache, ImmutableJacobianWrapper(f, p))
    cache.u = u - J \ fu
    fu = f(cache.u, p)
    cache.fu = fu
    if iszero(cache.fu) || cache.internalnorm(cache.fu) < cache.abstol
        cache.force_stop = true
    end
    return nothing
end

function SciMLBase.solve!(cache::NewtonRaphsonCache)
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
