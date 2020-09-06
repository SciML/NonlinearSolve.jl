struct Falsi <: AbstractBracketingAlgorithm
end

function alg_cache(alg::Falsi, left, right, p, ::Val{true})
  nothing
end

function alg_cache(alg::Falsi, left, right, p, ::Val{false})
  nothing
end

function perform_step!(solver, alg::Falsi, cache)
  @unpack f, p, left, right, fl, fr = solver

  fzero = zero(fl)
  fl * fr > fzero && error("Bracket became non-containing in between iterations. This could mean that "
  + "input function crosses the x axis multiple times. Bisection is not the right method to solve this.")

  mid = (fr * left - fl * right) / (fr - fl)
  
  if right == mid || right == mid
    solver.force_stop = true
    solver.retcode = :FloatingPointLimit
    return nothing
  end
  
  fm = f(mid, p)

  if iszero(fm)
    # todo: phase 2 bisection similar to the raw method
    solver.force_stop = true
    solver.left = mid
    solver.fl = fm
    solver.retcode = :ExactSolutionAtLeft
  else
    if sign(fm) == sign(fl)
      solver.left = mid
      solver.fl = fm
    else
      solver.right = mid
      solver.fr = fm
    end
  end
  return nothing
end
