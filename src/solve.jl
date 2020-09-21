function DiffEqBase.solve(prob::NonlinearProblem,
                          alg::AbstractNonlinearSolveAlgorithm, args...;
                          kwargs...)
  solver = DiffEqBase.init(prob, alg, args...; kwargs...)
  solve!(solver)
  return solver.sol
end

function DiffEqBase.init(prob::NonlinearProblem{uType, iip}, alg::AbstractBracketingAlgorithm, args...;
    alias_u0 = false,
    maxiters = 1000,
    immutable = (prob.u0 isa StaticArray || prob.u0 isa Number),
    kwargs...
  ) where {uType, iip}

  if !(prob.u0 isa Tuple)
    error("You need to pass a tuple of u0 in bracketing algorithms.")
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

  sol = build_solution(left, Val(iip))
  if immutable
    return BracketingImmutableSolver(1, f, alg, left, right, fl, fr, p, cache, false, maxiters, :Default, sol)
  else
    return BracketingSolver(1, f, alg, left, right, fl, fr, p, cache, false, maxiters, :Default, sol)
  end
end

function DiffEqBase.init(prob::NonlinearProblem{uType, iip}, alg::AbstractNewtonAlgorithm, args...;
    alias_u0 = false,
    maxiters = 1000,
    immutable = (prob.u0 isa StaticArray || prob.u0 isa Number),
    tol = 1e-6,
    internalnorm = Base.Fix2(DiffEqBase.ODE_DEFAULT_NORM, nothing),
    kwargs...
  ) where {uType, iip}

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


  sol = build_newton_solution(u, Val(iip))
  if immutable
    return NewtonImmutableSolver(1, f, alg, u, fu, p, nothing, false, maxiters, internalnorm, :Default, tol, sol)
  else
    cache = alg_cache(alg, f, u, p, Val(iip))
    return NewtonSolver(1, f, alg, u, fu, p, cache, false, maxiters, internalnorm, :Default, tol, sol)
  end
end

function DiffEqBase.solve!(solver::AbstractNonlinearSolver)
  # sync_residuals!(solver)
  mic_check!(solver)
  while !solver.force_stop && solver.iter < solver.maxiters
    perform_step!(solver, solver.alg, solver.cache)
    solver.iter += 1
    # sync_residuals!(solver)
  end
  if solver.iter == solver.maxiters
    solver.retcode = :MaxitersExceeded
  end
  solver = set_solution(solver)
  return solver.sol
end

function DiffEqBase.solve!(solver::AbstractImmutableNonlinearSolver)
  # sync_residuals!(solver)
  solver = mic_check(solver)
  while !solver.force_stop && solver.iter < solver.maxiters
    solver = perform_step(solver, solver.alg)
    @set! solver.iter += 1
    # sync_residuals!(solver)
  end
  if solver.iter == solver.maxiters
    @set! solver.retcode = :MaxitersExceeded
  end
  sol = get_solution(solver)
  return sol
end

function mic_check!(solver::BracketingSolver)
  @unpack f, fl, fr = solver
  flr = fl * fr
  fzero = zero(flr)
  (flr > fzero) && error("Non bracketing interval passed in bracketing method.")
  if fl == fzero
    solver.force_stop = true
    solver.retcode = :ExactSolutionAtLeft
  elseif fr == fzero
    solver.force_stop = true
    solver.retcode = :ExactSolutionAtRight
  end
  nothing
end

function mic_check(solver::BracketingImmutableSolver)
  @unpack f, fl, fr = solver
  flr = fl * fr
  fzero = zero(flr)
  (flr > fzero) && error("Non bracketing interval passed in bracketing method.")
  if fl == fzero
    @set! solver.force_stop = true
    @set! solver.retcode = :ExactSolutionAtLeft
  elseif fr == fzero
    @set! solver.force_stop = true
    @set! solver.retcode = :ExactSolutionAtRight
  end
  solver
end

function mic_check!(solver::NewtonSolver)
  nothing
end

function mic_check(solver::NewtonImmutableSolver)
  solver
end

function check_for_exact_solution!(solver::BracketingSolver)
  @unpack fl, fr = solver
  fzero = zero(fl)
  if fl == fzero
    solver.retcode = :ExactSolutionAtLeft
    return true
  elseif fr == fzero
    solver.retcode = :ExactSolutionAtRight
    return true
  end
  return false
end

function set_solution(solver::BracketingSolver)
  sol = solver.sol
  @set! sol.left = solver.left
  @set! sol.right = solver.right
  @set! sol.retcode = solver.retcode
  @set! solver.sol = sol
  return solver
end

function get_solution(solver::BracketingImmutableSolver)
  return (left = solver.left, right = solver.right, retcode = solver.retcode)
end

function set_solution(solver::NewtonSolver)
  sol = solver.sol
  @set! sol.u = solver.u
  @set! sol.retcode = solver.retcode
  @set! solver.sol = sol
  return solver
end

function get_solution(solver::NewtonImmutableSolver)
  return (u = solver.u, retcode = solver.retcode)
end

function reinit!(solver::NewtonSolver, prob::NonlinearProblem{uType, true}) where {uType}
  @. solver.u = prob.u0
  solver.iter = 1
  solver.force_stop = false
end

function reinit!(solver::NewtonSolver, prob::NonlinearProblem{uType, false}) where {uType}
  solver.u = prob.u0
  solver.iter = 1
  solver.force_stop = false
end
