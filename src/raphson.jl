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

mutable struct NewtonRaphsonCache{ufType, L, jType, uType, JC}
    uf::ufType
    linsolve::L
    J::jType
    du1::uType
    jac_config::JC
end

function dolinsolve(precs::P, linsolve; A = nothing, linu = nothing, b = nothing,
                    du = nothing, u = nothing, p = nothing, t = nothing,
                    weight = nothing, solverdata = nothing,
                    reltol = nothing) where {P}
    A !== nothing && (linsolve = LinearSolve.set_A(linsolve, A))
    b !== nothing && (linsolve = LinearSolve.set_b(linsolve, b))
    linu !== nothing && (linsolve = LinearSolve.set_u(linsolve, linu))

    Plprev = linsolve.Pl isa LinearSolve.ComposePreconditioner ? linsolve.Pl.outer :
             linsolve.Pl
    Prprev = linsolve.Pr isa LinearSolve.ComposePreconditioner ? linsolve.Pr.outer :
             linsolve.Pr

    _Pl, _Pr = precs(linsolve.A, du, u, p, nothing, A !== nothing, Plprev, Prprev,
                     solverdata)
    if (_Pl !== nothing || _Pr !== nothing)
        _weight = weight === nothing ?
                  (linsolve.Pr isa Diagonal ? linsolve.Pr.diag : linsolve.Pr.inner.diag) :
                  weight
        Pl, Pr = wrapprecs(_Pl, _Pr, _weight)
        linsolve = LinearSolve.set_prec(linsolve, Pl, Pr)
    end

    linres = if reltol === nothing
        solve(linsolve; reltol)
    else
        solve(linsolve; reltol)
    end

    return linres
end

function wrapprecs(_Pl, _Pr, weight)
    if _Pl !== nothing
        Pl = LinearSolve.ComposePreconditioner(LinearSolve.InvPreconditioner(Diagonal(_vec(weight))),
                                               _Pl)
    else
        Pl = LinearSolve.InvPreconditioner(Diagonal(_vec(weight)))
    end

    if _Pr !== nothing
        Pr = LinearSolve.ComposePreconditioner(Diagonal(_vec(weight)), _Pr)
    else
        Pr = Diagonal(_vec(weight))
    end
    Pl, Pr
end

function alg_cache(alg::NewtonRaphson, f, u, p, ::Val{true})
    uf = JacobianWrapper(f, p)
    J = false .* u .* u'

    linprob = LinearProblem(W, _vec(zero(u)); u0 = _vec(zero(u)))
    Pl, Pr = wrapprecs(alg.precs(W, nothing, u, p, nothing, nothing, nothing, nothing,
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
    NewtonRaphsonCache(uf, linsolve, J, du1, jac_config)
end

function alg_cache(alg::NewtonRaphson, f, u, p, ::Val{false})
    nothing
end

function perform_step(solver::NewtonImmutableSolver, alg::NewtonRaphson, ::Val{true})
    @unpack u, fu, f, p, cache = solver
    @unpack J, linsolve, du1 = cache
    calc_J!(J, solver, cache)
    # u = u - J \ fu
    linsolve = dolinsolve(alg.precs, solver.linsolve, A = J, b = fu, u = du1,
                          p = p, reltol = solver.tol)
    @set! cache.linsolve = linsolve
    @. u = u - du1
    f(fu, u, p)

    if solver.internalnorm(solver.fu) < solver.tol
        @set! solver.force_stop = true
    end
    return solver
end

function perform_step(solver::NewtonImmutableSolver, alg::NewtonRaphson, ::Val{false})
    @unpack u, fu, f, p = solver
    J = calc_J(solver, ImmutableJacobianWrapper(f, p))
    @set! solver.u = u - J \ fu
    fu = f(solver.u, p)
    @set! solver.fu = fu
    if iszero(solver.fu) || solver.internalnorm(solver.fu) < solver.tol
        @set! solver.force_stop = true
    end
    return solver
end
