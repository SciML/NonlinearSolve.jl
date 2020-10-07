struct Bisection <: AbstractBracketingAlgorithm
  exact_left::Bool
  exact_right::Bool
end

function Bisection(;exact_left=false, exact_right=false)
  Bisection(exact_left, exact_right)
end

struct BisectionCache{uType}
  state::Int
  left::uType
  right::uType
end

function alg_cache(alg::Bisection, left, right, p, ::Val{true})
  BisectionCache(0, left, right)
end

function alg_cache(alg::Bisection, left, right, p, ::Val{false})
  BisectionCache(0, left, right)
end

function perform_step(solver::BracketingImmutableSolver, alg::Bisection, cache)
  @unpack f, p, left, right, fl, fr, cache = solver

  if cache.state == 0
    fzero = zero(fl)
    fl * fr > fzero && error("Bracket became non-containing in between iterations. This could mean that "
    + "input function crosses the x axis multiple times. Bisection is not the right method to solve this.")

    mid = (left + right) / 2
    
    if left == mid || right == mid
      @set! solver.force_stop = true
      @set! solver.retcode = :FloatingPointLimit
      return solver
    end
    
    fm = f(mid, p)

    if iszero(fm)
      if alg.exact_left
        @set! cache.state = 1
        @set! cache.right = mid
        @set! cache.left = mid
        @set! solver.cache = cache
      elseif alg.exact_right
        @set! solver.left = prevfloat_tdir(mid, left, right)
        solver = sync_residuals!(solver)
        @set! cache.state = 2
        @set! cache.left = mid
        @set! solver.cache = cache
      else
        @set! solver.left = prevfloat_tdir(mid, left, right)
        @set! solver.right = nextfloat_tdir(mid, left, right)
        solver = sync_residuals!(solver)
        @set! solver.force_stop = true
        return solver
      end
    else
      if sign(fm) == sign(fl)
        @set! solver.left = mid
        @set! solver.fl = fm
      else
        @set! solver.right = mid
        @set! solver.fr = fm
      end
    end
  elseif cache.state == 1
    mid = (left + cache.right) / 2
    
    if cache.right == mid || left == mid
      if alg.exact_right
        @set! cache.state = 2
        @set! solver.cache = cache
        return solver
      else
        @set! solver.right = nextfloat_tdir(mid, left, right)
        solver = sync_residuals!(solver)
        @set! solver.force_stop = true
        return solver
      end
    end
    
    fm = f(mid, p)

    if iszero(fm)
      @set! cache.right = mid
      @set! solver.cache = cache
    else
      @set! solver.left = mid
      @set! solver.fl = fm
    end
  else
    mid = (cache.left + right) / 2
    
    if right == mid || cache.left == mid
      @set! solver.force_stop = true
      return solver
    end
    
    fm = f(mid, p)

    if iszero(fm)
      @set! cache.left = mid
      @set! solver.cache = cache
    else
      @set! solver.right = mid
      @set! solver.fr = fm
    end
  end
  solver
end
