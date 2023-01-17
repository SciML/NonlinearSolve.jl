function scalar_nlsolve_ad(prob, alg, args...; kwargs...)
    f = prob.f
    p = value(prob.p)

    u0 = value(prob.u0)
    newprob = NonlinearProblem(f, u0, p; prob.kwargs...)

    sol = solve(newprob, alg, args...; kwargs...)

    uu = sol.u
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

function SciMLBase.solve(prob::NonlinearProblem{<:Union{Number, StaticArraysCore.SVector},
                                                iip,
                                                <:Dual{T, V, P}},
                         alg::Union{NewtonRaphson, TrustRegion},
                         args...; kwargs...) where {iip, T, V, P}
    sol, partials = scalar_nlsolve_ad(prob, alg, args...; kwargs...)
    return SciMLBase.build_solution(prob, alg, Dual{T, V, P}(sol.u, partials), sol.resid;
                                    retcode = sol.retcode)
end
function SciMLBase.solve(prob::NonlinearProblem{<:Union{Number, StaticArraysCore.SVector},
                                                iip,
                                                <:AbstractArray{<:Dual{T, V, P}}},
                         alg::Union{NewtonRaphson, TrustRegion}, args...; kwargs...) where {iip, T, V, P}
    sol, partials = scalar_nlsolve_ad(prob, alg, args...; kwargs...)
    return SciMLBase.build_solution(prob, alg, Dual{T, V, P}(sol.u, partials), sol.resid;
                                    retcode = sol.retcode)
end
