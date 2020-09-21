function DiffEqBase.solve(prob::NonlinearProblem{<:Number}, ::NewtonRaphson, args...; xatol = nothing, xrtol = nothing, maxiters = 1000, kwargs...)
  f = Base.Fix2(prob.f, prob.p)
  x = float(prob.u0)
  T = typeof(x)
  atol = xatol !== nothing ? xatol : oneunit(T) * (eps(one(T)))^(4//5)
  rtol = xrtol !== nothing ? xrtol : eps(one(T))^(4//5)

  xo = oftype(x, Inf)
  for i in 1:maxiters
    fx, dfx = value_derivative(f, x)
    iszero(fx) && return x
    Δx = dfx \ fx
    x -= Δx
    if isapprox(x, xo, atol=atol, rtol=rtol)
        return x
    end
    xo = x
  end
  return oftype(x, NaN)
end

function DiffEqBase.solve(prob::NonlinearProblem{<:Number}, ::Bisection, args...; maxiters = 1000, kwargs...)
  f = Base.Fix2(prob.f, prob.p)
  left, right = prob.u0
  fl, fr = f(left), f(right)

  if iszero(fl)
    return fl
  end

  i = 1
  if !iszero(fr)
    while i < maxiters
      mid = (left + right) / 2
      (mid == left || mid == right) && return left
      fm = f(mid)
      if iszero(fm)
        right = mid
        break
      end
      if sign(fl) == sign(fm)
        fl = fm
        left = mid
      else
        fr = fm
        right = mid
      end
      i += 1
    end
  end

  while i < maxiters
    mid = (left + right) / 2
    (mid == left || mid == right) && return left
    fm = f(mid)
    if iszero(fm)
      right = mid
      fr = fm
    else
      left = mid
      fl = fm
    end
    i += 1
  end

  return left
end
