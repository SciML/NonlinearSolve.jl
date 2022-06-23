function SciMLBase.solve(prob::NonlinearProblem,
                         alg::AbstractNonlinearSolveAlgorithm, args...;
                         kwargs...)
    solver = init(prob, alg, args...; kwargs...)
    return sol = solve!(solver)
end

function SciMLBase.init(prob::NonlinearProblem{uType,iip}, alg::AbstractBracketingAlgorithm,
                        args...;
                        alias_u0 = false,
                        maxiters = 1000,
                        kwargs...) where {uType,iip}
    if !(prob.u0 isa Tuple)
        error("You need to pass a tuple of u0 in bracketing algorithms.")
    end

    if eltype(prob.u0) isa AbstractArray
        error("Bracketing Algorithms work for scalar arguments only")
    end

    if alias_u0
        left, right = prob.u0
    else
        left, right = deepcopy(prob.u0)
    end
    f = prob.f
    p = prob.p
    fl = f(left, p)
    fr = f(right, p)
    cache = alg_cache(alg, left, right, p, Val(iip))
    return BracketingImmutableSolver(1, f, alg, left, right, fl, fr, p, false, maxiters, DEFAULT,
                                     cache, iip, prob)
end

function SciMLBase.init(prob::NonlinearProblem{uType,iip}, alg::AbstractNewtonAlgorithm, args...;
                        alias_u0 = false,
                        maxiters = 1000,
                        tol = 1e-6,
                        internalnorm = DEFAULT_NORM,
                        kwargs...) where {uType,iip}
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
    cache = alg_cache(alg, f, u, p, Val(iip))
    return NewtonImmutableSolver(1, f, alg, u, fu, p, false, maxiters, internalnorm, DEFAULT, tol,
                                 cache, iip, prob)
end

function SciMLBase.solve!(solver::AbstractImmutableNonlinearSolver)
    solver = mic_check(solver)
    while !solver.force_stop && solver.iter < solver.maxiters
        solver = perform_step(solver, solver.alg, Val(solver.iip))
        @set! solver.iter += 1
    end
    if solver.iter == solver.maxiters
        @set! solver.retcode = MAXITERS_EXCEED
    end
    if typeof(solver) <: NewtonImmutableSolver
        SciMLBase.build_solution(solver.prob, solver.alg, solver.u, solver.fu;
                                 retcode = Symbol(solver.retcode))
    else
        SciMLBase.build_solution(solver.prob, solver.alg, solver.left, solver.fl;
                                 retcode = Symbol(solver.retcode), left = solver.left,
                                 right = solver.right)
    end
end

"""
  mic_check(solver::AbstractImmutableNonlinearSolver)
  mic_check!(solver::AbstractNonlinearSolver)

Checks before running main solving iterations.
"""
function mic_check(solver::BracketingImmutableSolver)
    @unpack f, fl, fr = solver
    flr = fl * fr
    fzero = zero(flr)
    (flr > fzero) && error("Non bracketing interval passed in bracketing method.")
    if fl == fzero
        @set! solver.force_stop = true
        @set! solver.retcode = EXACT_SOLUTION_LEFT
    elseif fr == fzero
        @set! solver.force_stop = true
        @set! solver.retcode = EXACT_SOLUTION_RIGHT
    end
    return solver
end

function mic_check(solver::NewtonImmutableSolver)
    return solver
end

"""
  reinit!(solver, prob)

Reinitialize solver to the original starting conditions
"""
function SciMLBase.reinit!(solver::NewtonImmutableSolver,
                           prob::NonlinearProblem{uType,true}) where {uType}
    @. solver.u = prob.u0
    @set! solver.iter = 1
    @set! solver.force_stop = false
    return solver
end

function SciMLBase.reinit!(solver::NewtonImmutableSolver,
                           prob::NonlinearProblem{uType,false}) where {uType}
    @set! solver.u = prob.u0
    @set! solver.iter = 1
    @set! solver.force_stop = false
    return solver
end
