function SciMLBase.solve(
        prob::NonlinearProblem{<:Union{Number, <:AbstractArray},
            iip, <:Union{<:Dual{T, V, P}, <:AbstractArray{<:Dual{T, V, P}}}},
        alg::AbstractSimpleNonlinearSolveAlgorithm, args...; kwargs...) where {T, V, P, iip}
    sol, partials = __nlsolve_ad(prob, alg, args...; kwargs...)
    dual_soln = __nlsolve_dual_soln(sol.u, partials, prob.p)
    return SciMLBase.build_solution(
        prob, alg, dual_soln, sol.resid; sol.retcode, sol.stats,
        sol.original)
end

for algType in (Bisection, Brent, Alefeld, Falsi, ITP, Ridder)
    @eval begin
        function SciMLBase.solve(
                prob::IntervalNonlinearProblem{uType, iip,
                    <:Union{<:Dual{T, V, P}, <:AbstractArray{<:Dual{T, V, P}}}},
                alg::$(algType), args...; kwargs...) where {uType, T, V, P, iip}
            sol, partials = __nlsolve_ad(prob, alg, args...; kwargs...)
            dual_soln = __nlsolve_dual_soln(sol.u, partials, prob.p)
            return SciMLBase.build_solution(prob, alg, dual_soln, sol.resid; sol.retcode,
                sol.stats, sol.original, left = Dual{T, V, P}(sol.left, partials),
                right = Dual{T, V, P}(sol.right, partials))
        end
    end
end

function __nlsolve_ad(prob, alg, args...; kwargs...)
    p = value(prob.p)
    if prob isa IntervalNonlinearProblem
        tspan = value.(prob.tspan)
        newprob = IntervalNonlinearProblem(prob.f, tspan, p; prob.kwargs...)
    else
        u0 = value(prob.u0)
        newprob = NonlinearProblem(prob.f, u0, p; prob.kwargs...)
    end

    sol = solve(newprob, alg, args...; kwargs...)

    uu = sol.u
    f_p = __nlsolve_∂f_∂p(prob, prob.f, uu, p)
    f_x = __nlsolve_∂f_∂u(prob, prob.f, uu, p)

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

@inline function __nlsolve_∂f_∂p(prob, f::F, u, p) where {F}
    if isinplace(prob)
        __f = p -> begin
            du = similar(u, promote_type(eltype(u), eltype(p)))
            f(du, u, p)
            return du
        end
    else
        __f = Base.Fix1(f, u)
    end
    if p isa Number
        return __reshape(ForwardDiff.derivative(__f, p), :, 1)
    elseif u isa Number
        return __reshape(ForwardDiff.gradient(__f, p), 1, :)
    else
        return ForwardDiff.jacobian(__f, p)
    end
end

@inline function __nlsolve_∂f_∂u(prob, f::F, u, p) where {F}
    if isinplace(prob)
        du = similar(u)
        __f = (du, u) -> f(du, u, p)
        ForwardDiff.jacobian(__f, du, u)
    else
        __f = Base.Fix2(f, p)
        if u isa Number
            return ForwardDiff.derivative(__f, u)
        else
            return ForwardDiff.jacobian(__f, u)
        end
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
