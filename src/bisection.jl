struct Bisection <: AbstractBracketingAlgorithm
end

mutable struct BisectionCache{uType}
  state::UInt8
  left::uType
  right::uType
end

function alg_cache(alg::Bisection, left, right, p, ::Val{true})
  BisectionCache(UInt8(0), left, right)
end

function alg_cache(alg::Bisection, left, right, p, ::Val{false})
  BisectionCache(UInt8(0), left, right)
end

"""
  bisection(f, tup ; maxiters=1000)

Uses bisection method to find the root of the function `f` between a tuple `tup` of values.
"""
function bisection(f, tup ; maxiters=1000)
  x0, x1 = tup
  fx0, fx1 = f(x0), f(x1)
  fx0x1 = fx0 * fx1
  fzero = zero(fx0x1)

  (fx0x1 > fzero) && error("Non bracketing interval passed in bisection method.")
  # NOTE: fx0x1 = 0 can mean that both fx0 and fx1 are very small and multiplication of them 
  # could be less than the smallest float, hence could be zero.

  fx0 == fzero && return x0 # should replace with some tolerance compare i.e. ≈
  fx1 == fzero && return x1

  left = x0
  right = x1

  iter = 0
  while true
    iter += 1
    
    if iter == maxiters
      return left
    end

    fl = f(left)
    fr = f(right)

    fl * fr >= fzero && error("Bracket became non-containing in between iterations. This could mean that "
    + "input function crosses the x axis multiple times. Bisection is not the right method to solve this.")

    mid = (left + right) / 2.0
    fm = f(mid)
    if iszero(fm)
      # we are in the region of zero, inner loop
      right = mid
      while true
        iter += 1
        
        if iter == maxiters
          return left
        end
        
        mid = (left + right) / 2.0
        (left == mid || right == mid) && return left
        fm = f(mid)

        if iszero(fm)
          if !iszero(f(prevfloat_tdir(mid, x0, x1)))
            return prevfloat_tdir(mid, x0, x1)
          end
          right = mid
        else
          left = mid
        end

      end
    end

    (left == mid || right == mid) && return left
    if sign(fm) == sign(fl)
      left = mid
    else
      right = mid
    end
  end
end

function perform_step!(solver, alg::Bisection, cache)
  @unpack f, p, left, right, fl, fr = solver

  if cache.state == 0
    fzero = zero(fl)
    fl * fr > fzero && error("Bracket became non-containing in between iterations. This could mean that "
    + "input function crosses the x axis multiple times. Bisection is not the right method to solve this.")

    mid = (left + right) / 2.0
    
    if left == mid || right == mid
      solver.force_stop = true
      solver.retcode = :FloatingPointLimit
      return
    end
    
    fm = f(mid, p)

    if iszero(fm)
      cache.state = 1
      cache.right = mid
      cache.left = mid
    else
      if sign(fm) == sign(fl)
        solver.left = mid
        solver.fl = fm
      else
        solver.right = mid
        solver.fr = fm
      end
    end
  elseif cache.state == 1
    mid = (left + cache.right) / 2.0
    
    if cache.right == mid || left == mid
      cache.state = 2
      return
    end
    
    fm = f(mid, p)

    if iszero(fm)
      cache.right = mid
    else
      solver.left = mid
      solver.fl = fm
    end
  else
    mid = (cache.left + right) / 2.0
    
    if right == mid || cache.left == mid
      solver.force_stop = true
      return
    end
    
    fm = f(mid, p)

    if iszero(fm)
      cache.left = mid
    else
      solver.right = mid
      solver.fr = fm
    end
  end
end
