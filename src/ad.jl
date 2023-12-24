function SciMLBase.solve(prob::NonlinearProblem{<:Union{Number, <:AbstractArray},
            iip, <:Union{<:Dual{T, V, P}, <:AbstractArray{<:Dual{T, V, P}}}},
        alg::Union{Nothing, AbstractNonlinearAlgorithm}, args...;
        kwargs...) where {T, V, P, iip}
    sol, partials = __nlsolve_ad(prob, alg, args...; kwargs...)
    dual_soln = __nlsolve_dual_soln(sol.u, partials, prob.p)
    return SciMLBase.build_solution(prob, alg, dual_soln, sol.resid; sol.retcode)
end

# Differentiate Out-of-Place Nonlinear Root Finding Problems
function __nlsolve_ad(prob::NonlinearProblem{uType, false}, alg, args...;
        kwargs...) where {uType}
    p = value(prob.p)
    newprob = NonlinearProblem(prob.f, value(prob.u0), p; prob.kwargs...)

    sol = solve(newprob, alg, args...; kwargs...)

    uu = sol.u
    f_p = __nlsolve_∂f_∂p(prob.f, uu, p)
    f_x = __nlsolve_∂f_∂u(prob.f, uu, p)

    z_arr = -f_x \ f_p

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

@inline function __nlsolve_∂f_∂p(f::F, u, p) where {F}
    __f = Base.Fix1(f, u)
    if p isa Number
        return __reshape(ForwardDiff.derivative(__f, p), :, 1)
    elseif u isa Number
        return __reshape(ForwardDiff.gradient(__f, p), 1, :)
    else
        return ForwardDiff.jacobian(__f, p)
    end
end

@inline function __nlsolve_∂f_∂u(f::F, u, p) where {F}
    __f = Base.Fix2(f, p)
    if u isa Number
        return ForwardDiff.derivative(__f, u)
    else
        return ForwardDiff.jacobian(__f, u)
    end
end

@inline function __nlsolve_dual_soln(u::Number, partials,
        ::Union{<:AbstractArray{<:Dual{T, V, P}}, Dual{T, V, P}}) where {T, V, P}
    return Dual{T, V, P}(u, partials)
end

@inline function __nlsolve_dual_soln(u::AbstractArray, partials,
        ::Union{<:AbstractArray{<:Dual{T, V, P}}, Dual{T, V, P}}) where {T, V, P}
    _partials = _restructure(u, partials)
    return map(((uᵢ, pᵢ),) -> Dual{T, V, P}(uᵢ, pᵢ), zip(u, _partials))
end
