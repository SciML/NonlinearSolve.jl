function solve(prob::NonlinearProblem{<:Number}, ::NewtonRaphson, args...; xatol = nothing, xrtol = nothing, maxiters = 1000, kwargs...)
  f = Base.Fix2(prob.f, prob.p)
  x = float(prob.u0)
  T = typeof(x)
  atol = xatol !== nothing ? xatol : oneunit(T) * (eps(one(T)))^(4//5)
  rtol = xrtol !== nothing ? xrtol : eps(one(T))^(4//5)

  xo = oftype(x, Inf)
  for i in 1:maxiters
    fx, dfx = value_derivative(f, x)
    iszero(fx) && return NewtonSolution(x, DEFAULT)
    Δx = dfx \ fx
    x -= Δx
    if isapprox(x, xo, atol=atol, rtol=rtol)
        return NewtonSolution(x, DEFAULT)
    end
    xo = x
  end
  return NewtonSolution(x, MAXITERS_EXCEED)
end

function scalar_nlsolve_ad(prob, alg, args...; kwargs...)
  f = prob.f
  p = value(prob.p)
  u0 = value(prob.u0)

  newprob = NonlinearProblem(f, u0, p; prob.kwargs...)
  sol = solve(newprob, alg, args...; kwargs...)

  uu = getsolution(sol)
  if p isa Number
    f_p = ForwardDiff.derivative(Base.Fix1(f, uu), p)
  else
    f_p = ForwardDiff.gradient(Base.Fix1(f, uu), p)
  end

  f_x = ForwardDiff.derivative(Base.Fix2(f, p), uu)
  pp = prob.p
  sumfun = let f_x′ = -f_x
    ((fp, p),) -> (fp / f_x′) * ForwardDiff.partials(p)
  end
  partials = sum(sumfun, zip(f_p, pp))
  return sol, partials
end

function solve(prob::NonlinearProblem{<:Number, iip, <:Dual{T,V,P}}, alg::NewtonRaphson, args...; kwargs...) where {iip, T, V, P}
  sol, partials = scalar_nlsolve_ad(prob, alg, args...; kwargs...)
  return NewtonSolution(Dual{T,V,P}(sol.u, partials), sol.retcode)
end
function solve(prob::NonlinearProblem{<:Number, iip, <:AbstractArray{<:Dual{T,V,P}}}, alg::NewtonRaphson, args...; kwargs...) where {iip, T, V, P}
  sol, partials = scalar_nlsolve_ad(prob, alg, args...; kwargs...)
  return NewtonSolution(Dual{T,V,P}(sol.u, partials), sol.retcode)
end

# avoid ambiguities
for Alg in [Bisection]
  @eval function solve(prob::NonlinearProblem{uType, iip, <:Dual{T,V,P}}, alg::$Alg, args...; kwargs...) where {uType, iip, T, V, P}
    sol, partials = scalar_nlsolve_ad(prob, alg, args...; kwargs...)
    return BracketingSolution(Dual{T,V,P}(sol.left, partials), Dual{T,V,P}(sol.right, partials), sol.retcode)
  end
  @eval function solve(prob::NonlinearProblem{uType, iip, <:AbstractArray{<:Dual{T,V,P}}}, alg::$Alg, args...; kwargs...) where {uType, iip, T, V, P}
    sol, partials = scalar_nlsolve_ad(prob, alg, args...; kwargs...)
    return BracketingSolution(Dual{T,V,P}(sol.left, partials), Dual{T,V,P}(sol.right, partials), sol.retcode)
  end
end

function solve(prob::NonlinearProblem, ::Bisection, args...; maxiters = 1000, kwargs...)
  f = Base.Fix2(prob.f, prob.p)
  left, right = prob.u0
  fl, fr = f(left), f(right)

  if iszero(fl)
    return BracketingSolution(left, right, EXACT_SOLUTION_LEFT)
  end

  i = 1
  if !iszero(fr)
    while i < maxiters
      mid = (left + right) / 2
      (mid == left || mid == right) && return BracketingSolution(left, right, FLOATING_POINT_LIMIT)
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
    (mid == left || mid == right) && return BracketingSolution(left, right, FLOATING_POINT_LIMIT)
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

  return BracketingSolution(left, right, MAXITERS_EXCEED)
end

function solve(prob::NonlinearProblem, ::Falsi, args...; maxiters = 1000, kwargs...)
  f = Base.Fix2(prob.f, prob.p)
  left, right = prob.u0
  fl, fr = f(left), f(right)

  if iszero(fl)
    return BracketingSolution(left, right, EXACT_SOLUTION_LEFT)
  end

  i = 1
  if !iszero(fr)
    while i < maxiters
      if nextfloat_tdir(left, prob.u0...) == right
        return BracketingSolution(left, right, FLOATING_POINT_LIMIT)
      end
      mid = (fr * left - fl * right) / (fr - fl)
      for i in 1:10
        mid = max(left, prevfloat_tdir(mid, prob.u0...))
      end
      if mid == right || mid == left
        break
      end
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
    (mid == left || mid == right) && return BracketingSolution(left, right, FLOATING_POINT_LIMIT)
    fm = f(mid)
    if iszero(fm)
      right = mid
      fr = fm
    elseif sign(fm) == sign(fl)
      left = mid
      fl = fm
    else
      right = mid
      fr = fm
    end
    i += 1
  end

  return BracketingSolution(left, right, MAXITERS_EXCEED)
end
