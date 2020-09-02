function DiffEqBase.__solve(prob::NonlinearProblem,
                            alg::AbstractNonlinearSolveAlgorithm, args...;
                            kwargs...)
  solver = DiffEqBase.__init(prob, alg, args...; kwargs...)
  solve!(solver)
  return solver.sol
end

function DiffEqBase.__init(prob::NonlinearProblem{uType, iip}, alg::AbstractBracketingAlgorithm, args...;
    alias_u0 = false,
    maxiters = 1000,
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
  return BracketingSolver(1, f, alg, left, right, fl, fr, p, cache, false, maxiters, :Default, sol)
end

function DiffEqBase.solve!(solver::BracketingSolver)
  # sync_residuals!(solver)
  mic_check!(solver)
  while !solver.force_stop && solver.iter < solver.maxiters
    if check_for_exact_solution!(solver)
      break
    else
      perform_step!(solver, solver.alg, solver.cache)
      solver.iter += 1
    end
    # sync_residuals!(solver)
  end
  if solver.iter == solver.maxiters
    solver.retcode = :MaxitersExceeded
  end
  set_solution!(solver)
  return solver.sol
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

function set_solution!(solver)
  solver.sol.left = solver.left
  solver.sol.right = solver.right
  solver.sol.retcode = solver.retcode
end
