module NonlinearSolveBaseForwardDiffExt

using ADTypes: ADTypes, AutoForwardDiff, AutoPolyesterForwardDiff
using ArrayInterface: ArrayInterface
using CommonSolve: CommonSolve, solve, solve!, init
using DifferentiationInterface: DifferentiationInterface
using FastClosures: @closure
using ForwardDiff: ForwardDiff, Dual, pickchunksize
using SciMLBase: SciMLBase, AbstractNonlinearProblem, IntervalNonlinearProblem,
    NonlinearProblem, NonlinearLeastSquaresProblem, remake

using NonlinearSolveBase: NonlinearSolveBase, ImmutableNonlinearProblem, Utils, InternalAPI,
    NonlinearSolvePolyAlgorithm, NonlinearSolveForwardDiffCache

const DI = DifferentiationInterface

const GENERAL_SOLVER_TYPES = [
    Nothing, NonlinearSolvePolyAlgorithm,
]

const DualNonlinearProblem = NonlinearProblem{
    <:Union{Number, <:AbstractArray}, iip,
    <:Union{<:Dual{T, V, P}, <:AbstractArray{<:Dual{T, V, P}}},
} where {iip, T, V, P}
const DualNonlinearLeastSquaresProblem = NonlinearLeastSquaresProblem{
    <:Union{Number, <:AbstractArray}, iip,
    <:Union{<:Dual{T, V, P}, <:AbstractArray{<:Dual{T, V, P}}},
} where {iip, T, V, P}
const DualAbstractNonlinearProblem = Union{
    DualNonlinearProblem, DualNonlinearLeastSquaresProblem,
}

function NonlinearSolveBase.additional_incompatible_backend_check(
        prob::AbstractNonlinearProblem, ::Union{AutoForwardDiff, AutoPolyesterForwardDiff}
    )
    return !ForwardDiff.can_dual(eltype(prob.u0))
end

Utils.value(::Type{Dual{T, V, N}}) where {T, V, N} = V
Utils.value(x::Dual) = ForwardDiff.value(x)
Utils.value(x::AbstractArray{<:Dual}) = Utils.value.(x)

function NonlinearSolveBase.nonlinearsolve_forwarddiff_solve(
        prob::Union{
            IntervalNonlinearProblem, NonlinearProblem,
            ImmutableNonlinearProblem, NonlinearLeastSquaresProblem,
        },
        alg, args...; kwargs...
    )
    p = Utils.value(prob.p)
    if prob isa IntervalNonlinearProblem
        tspan = Utils.value.(prob.tspan)
        newprob = IntervalNonlinearProblem(prob.f, tspan, p; prob.kwargs...)
    else
        newprob = remake(prob; p, u0 = Utils.value(prob.u0))
    end

    sol = solve(newprob, alg, args...; kwargs...)
    uu = sol.u

    fn = prob isa NonlinearLeastSquaresProblem ?
        NonlinearSolveBase.nlls_generate_vjp_function(prob, sol, uu) : prob.f

    Jₚ = NonlinearSolveBase.nonlinearsolve_∂f_∂p(prob, fn, uu, p)
    Jᵤ = NonlinearSolveBase.nonlinearsolve_∂f_∂u(prob, fn, uu, p)
    z = -Jᵤ \ Jₚ
    pp = prob.p
    sumfun = ((z, p),) -> map(Base.Fix2(*, ForwardDiff.partials(p)), z)

    if uu isa Number
        partials = sum(sumfun, zip(z, pp))
    elseif p isa Number
        partials = sumfun((z, pp))
    else
        partials = sum(sumfun, zip(eachcol(z), pp))
    end

    return sol, partials
end

function NonlinearSolveBase.nonlinearsolve_∂f_∂p(prob, f::F, u, p) where {F}
    if SciMLBase.isinplace(prob)
        f2 = @closure p -> begin
            du = Utils.safe_similar(u, promote_type(eltype(u), eltype(p)))
            f(du, u, p)
            return du
        end
    else
        f2 = Base.Fix1(f, u)
    end
    if p isa Number
        return Utils.safe_reshape(ForwardDiff.derivative(f2, p), :, 1)
    elseif u isa Number
        return Utils.safe_reshape(ForwardDiff.gradient(f2, p), 1, :)
    else
        return ForwardDiff.jacobian(f2, p)
    end
end

function NonlinearSolveBase.nonlinearsolve_∂f_∂u(prob, f::F, u, p) where {F}
    if SciMLBase.isinplace(prob)
        return ForwardDiff.jacobian(
            @closure((du, u) -> f(du, u, p)), Utils.safe_similar(u), u
        )
    end
    u isa Number && return ForwardDiff.derivative(Base.Fix2(f, p), u)
    return ForwardDiff.jacobian(Base.Fix2(f, p), u)
end

function NonlinearSolveBase.nonlinearsolve_dual_solution(
        u::Number, partials,
        ::Union{<:AbstractArray{<:Dual{T, V, P}}, Dual{T, V, P}}
    ) where {T, V, P}
    return Dual{T, V, P}(u, partials)
end

function NonlinearSolveBase.nonlinearsolve_dual_solution(
        u::AbstractArray, partials,
        ::Union{<:AbstractArray{<:Dual{T, V, P}}, Dual{T, V, P}}
    ) where {T, V, P}
    return map(((uᵢ, pᵢ),) -> Dual{T, V, P}(uᵢ, pᵢ), zip(u, Utils.restructure(u, partials)))
end

for algType in GENERAL_SOLVER_TYPES
    @eval function SciMLBase.__solve(
            prob::DualAbstractNonlinearProblem, alg::$(algType), args...; kwargs...
        )
        sol,
            partials = NonlinearSolveBase.nonlinearsolve_forwarddiff_solve(
            prob, alg, args...; kwargs...
        )
        dual_soln = NonlinearSolveBase.nonlinearsolve_dual_solution(sol.u, partials, prob.p)
        return SciMLBase.build_solution(
            prob, alg, dual_soln, sol.resid; sol.retcode, sol.stats, sol.original
        )
    end
end

function InternalAPI.reinit!(
        cache::NonlinearSolveForwardDiffCache, args...;
        p = cache.p, u0 = NonlinearSolveBase.get_u(cache.cache), kwargs...
    )
    InternalAPI.reinit!(
        cache.cache; p = NonlinearSolveBase.nodual_value(p),
        u0 = NonlinearSolveBase.nodual_value(u0), kwargs...
    )
    cache.p = p
    cache.values_p = NonlinearSolveBase.nodual_value(p)
    cache.partials_p = ForwardDiff.partials(p)
    return cache
end

for algType in GENERAL_SOLVER_TYPES
    @eval function SciMLBase.__init(
            prob::DualAbstractNonlinearProblem, alg::$(algType), args...; kwargs...
        )
        p = NonlinearSolveBase.nodual_value(prob.p)
        newprob = SciMLBase.remake(prob; u0 = NonlinearSolveBase.nodual_value(prob.u0), p)
        cache = init(newprob, alg, args...; kwargs...)
        return NonlinearSolveForwardDiffCache(
            cache, newprob, alg, prob.p, p, ForwardDiff.partials(prob.p)
        )
    end
end

function CommonSolve.solve!(cache::NonlinearSolveForwardDiffCache)
    sol = solve!(cache.cache)
    prob = cache.prob
    uu = sol.u

    fn = prob isa NonlinearLeastSquaresProblem ?
        NonlinearSolveBase.nlls_generate_vjp_function(prob, sol, uu) : prob.f

    Jₚ = NonlinearSolveBase.nonlinearsolve_∂f_∂p(prob, fn, uu, cache.values_p)
    Jᵤ = NonlinearSolveBase.nonlinearsolve_∂f_∂u(prob, fn, uu, cache.values_p)

    z_arr = -Jᵤ \ Jₚ

    sumfun = ((z, p),) -> map(zᵢ -> zᵢ * ForwardDiff.partials(p), z)
    if cache.p isa Number
        partials = sumfun((z_arr, cache.p))
    else
        partials = sum(sumfun, zip(eachcol(z_arr), cache.p))
    end

    dual_soln = NonlinearSolveBase.nonlinearsolve_dual_solution(sol.u, partials, cache.p)
    return SciMLBase.build_solution(
        prob, cache.alg, dual_soln, sol.resid; sol.retcode, sol.stats, sol.original
    )
end

NonlinearSolveBase.nodual_value(x) = x
NonlinearSolveBase.nodual_value(x::Dual) = ForwardDiff.value(x)
NonlinearSolveBase.nodual_value(x::AbstractArray{<:Dual}) = map(ForwardDiff.value, x)

@inline NonlinearSolveBase.pickchunksize(x) = pickchunksize(length(x))
@inline NonlinearSolveBase.pickchunksize(x::Int) = ForwardDiff.pickchunksize(x)

end
