struct Falsi <: AbstractBracketingAlgorithm
end

function alg_cache(alg::Falsi, left, right, p, ::Val{true})
  nothing
end

function alg_cache(alg::Falsi, left, right, p, ::Val{false})
  nothing
end

function perform_step(solver, alg::Falsi, cache)
  @unpack f, p, left, right, fl, fr = solver

  fzero = zero(fl)
  fl * fr > fzero && error("Bracket became non-containing in between iterations. This could mean that "
  + "input function crosses the x axis multiple times. Bisection is not the right method to solve this.")

  mid = (fr * left - fl * right) / (fr - fl)
  
  if right == mid || right == mid
    @set! solver.force_stop = true
    @set! solver.retcode = :FloatingPointLimit
    return solver
  end
  
  fm = f(mid, p)

  if iszero(fm)
    # todo: phase 2 bisection similar to the raw method
    @set! solver.force_stop = true
    @set! solver.left = mid
    @set! solver.fl = fm
    @set! solver.retcode = :ExactSolutionAtLeft
  else
    if sign(fm) == sign(fl)
      @set! solver.left = mid
      @set! solver.fl = fm
    else
      @set! solver.right = mid
      @set! solver.fr = fm
    end
  end
  return solver
end
