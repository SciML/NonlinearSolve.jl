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

function solve(prob::NonlinearProblem{<:Number, iip, <:ForwardDiff.Dual{T,V,P}}, alg::NewtonRaphson, args...; kwargs...) where {uType, iip, T, V, P}
  f = prob.f
  p = ForwardDiff.value(prob.p)
  u0 = ForwardDiff.value(prob.u0)
  newprob = NonlinearProblem(f, u0, p; prob.kwargs...)
  sol = solve(newprob, alg, args...; kwargs...)
  f_p = ForwardDiff.derivative(Base.Fix1(f, sol.u), p)
  f_x = ForwardDiff.derivative(Base.Fix2(f, p), sol.u)
  partials = (-f_p / f_x) * ForwardDiff.partials(prob.p)
  return NewtonSolution(ForwardDiff.Dual{T,V,P}(sol.u, partials), sol.retcode)
end

function solve(prob::NonlinearProblem{uType, iip, <:ForwardDiff.Dual{T,V,P}}, alg::Bisection, args...; kwargs...) where {uType, iip, T, V, P}
  prob_nodual = NonlinearProblem(prob.f, prob.u0, ForwardDiff.value(prob.p); prob.kwargs...)
  sol = solve(prob_nodual, alg, args...; kwargs...)
  # f, x and p always satisfy
  # f(x, p) = 0
  # dx * f_x(x, p) + dp * f_p(x, p) = 0
  # dx / dp = - f_p(x, p) / f_x(x, p)
  f_p = (p) -> prob.f(sol.left, p)
  f_x = (x) -> prob.f(x, ForwardDiff.value(prob.p))
  d_p = ForwardDiff.derivative(f_p, ForwardDiff.value(prob.p))
  d_x = ForwardDiff.derivative(f_x, sol.left)
  partials = - d_p / d_x * ForwardDiff.partials(prob.p)
  return BracketingSolution(ForwardDiff.Dual{T,V,P}(sol.left, partials), ForwardDiff.Dual{T,V,P}(sol.right, partials), sol.retcode)
end

# still WIP
function solve(prob::NonlinearProblem{uType, iip, <:AbstractArray{<:ForwardDiff.Dual{T,V,P}, N}}, alg::Bisection, args...; kwargs...) where {uType, iip, T, V, P, N}
  p_nodual = ForwardDiff.value.(prob.p)
  prob_nodual = NonlinearProblem(prob.f, prob.u0, p_nodual; prob.kwargs...)
  sol = solve(prob_nodual, alg, args...; kwargs...)
  # f, x and p always satisfy
  # f(x, p) = 0
  # dx * f_x(x, p) + dp * f_p(x, p) = 0
  # dx / dp = - f_p(x, p) / f_x(x, p)
  f_p = (p) -> [ prob.f(sol.left, p) ]
  f_x = (x) -> prob.f(x, p_nodual)
  d_p = ForwardDiff.jacobian(f_p, p_nodual)
  d_x = ForwardDiff.derivative(f_x, sol.left)
  @. d_p =  - d_p / d_x
  @show ForwardDiff.partials.(prob.p)
  return ForwardDiff.Dual{T,V,P}(sol.left, d_p * ForwardDiff.partials.(prob.p))
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
