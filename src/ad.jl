function scalar_nlsolve_ad(prob, alg, args...; kwargs...)
    f = prob.f
    p = value(prob.p)
    u0 = value(prob.u0)
    newprob = NonlinearProblem(f, u0, p; prob.kwargs...)

    sol = solve(newprob, alg, args...; kwargs...)

    uu = sol.u
    f_p = scalar_nlsolve_∂f_∂p(f, uu, p)
    f_x = scalar_nlsolve_∂f_∂u(f, uu, p)

    z_arr = -inv(f_x) * f_p

    pp = prob.p
    sumfun = ((z, p),) -> map(zᵢ -> zᵢ * ForwardDiff.partials(p), z)
    if uu isa Number
        partials = sum(sumfun, zip(z_arr, pp))
    elseif p isa Number
        partials = sumfun((z_arr, pp))
    else
        partials = sum(sumfun, zip(eachcol(z_arr), pp))
    end

    return sol, partials
end

function SciMLBase.solve(prob::NonlinearProblem{<:Union{Number, SVector, <:AbstractArray},
        false, <:Dual{T, V, P}}, alg::AbstractNewtonAlgorithm, args...;
    kwargs...) where {T, V, P}
    sol, partials = scalar_nlsolve_ad(prob, alg, args...; kwargs...)
    dual_soln = scalar_nlsolve_dual_soln(sol.u, partials, prob.p)
    return SciMLBase.build_solution(prob, alg, dual_soln, sol.resid; sol.retcode)
end

function SciMLBase.solve(prob::NonlinearProblem{<:Union{Number, SVector, <:AbstractArray},
        false, <:AbstractArray{<:Dual{T, V, P}}}, alg::AbstractNewtonAlgorithm, args...;
    kwargs...) where {T, V, P}
    sol, partials = scalar_nlsolve_ad(prob, alg, args...; kwargs...)
    dual_soln = scalar_nlsolve_dual_soln(sol.u, partials, prob.p)
    return SciMLBase.build_solution(prob, alg, dual_soln, sol.resid; sol.retcode)
end

function scalar_nlsolve_∂f_∂p(f, u, p)
    ff = p isa Number ? ForwardDiff.derivative :
         (u isa Number ? ForwardDiff.gradient : ForwardDiff.jacobian)
    return ff(Base.Fix1(f, u), p)
end

function scalar_nlsolve_∂f_∂u(f, u, p)
    ff = u isa Number ? ForwardDiff.derivative : ForwardDiff.jacobian
    return ff(Base.Fix2(f, p), u)
end

function scalar_nlsolve_dual_soln(u::Number, partials,
    ::Union{<:AbstractArray{<:Dual{T, V, P}}, Dual{T, V, P}}) where {T, V, P}
    return Dual{T, V, P}(u, partials)
end

function scalar_nlsolve_dual_soln(u::AbstractArray, partials,
    ::Union{<:AbstractArray{<:Dual{T, V, P}}, Dual{T, V, P}}) where {T, V, P}
    return map(((uᵢ, pᵢ),) -> Dual{T, V, P}(uᵢ, pᵢ), zip(u, partials))
end
