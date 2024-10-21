module NonlinearSolveBaseForwardDiffExt

using ADTypes: ADTypes, AutoForwardDiff, AutoPolyesterForwardDiff
using ArrayInterface: ArrayInterface
using CommonSolve: solve
using DifferentiationInterface: DifferentiationInterface
using FastClosures: @closure
using ForwardDiff: ForwardDiff, Dual
using LinearAlgebra: mul!
using SciMLBase: SciMLBase, AbstractNonlinearProblem, IntervalNonlinearProblem,
                 NonlinearProblem, NonlinearLeastSquaresProblem, remake

using NonlinearSolveBase: NonlinearSolveBase, ImmutableNonlinearProblem, Utils

const DI = DifferentiationInterface

function NonlinearSolveBase.additional_incompatible_backend_check(
        prob::AbstractNonlinearProblem, ::Union{AutoForwardDiff, AutoPolyesterForwardDiff})
    return !ForwardDiff.can_dual(eltype(prob.u0))
end

Utils.value(::Type{Dual{T, V, N}}) where {T, V, N} = V
Utils.value(x::Dual) = Utils.value(ForwardDiff.value(x))
Utils.value(x::AbstractArray{<:Dual}) = Utils.value.(x)

function NonlinearSolveBase.nonlinearsolve_forwarddiff_solve(
        prob::Union{IntervalNonlinearProblem, NonlinearProblem, ImmutableNonlinearProblem},
        alg, args...; kwargs...)
    p = Utils.value(prob.p)
    if prob isa IntervalNonlinearProblem
        tspan = Utils.value.(prob.tspan)
        newprob = IntervalNonlinearProblem(prob.f, tspan, p; prob.kwargs...)
    else
        newprob = remake(prob; p, u0 = Utils.value(prob.u0))
    end

    sol = solve(newprob, alg, args...; kwargs...)

    uu = sol.u
    Jₚ = NonlinearSolveBase.nonlinearsolve_∂f_∂p(prob, prob.f, uu, p)
    Jᵤ = NonlinearSolveBase.nonlinearsolve_∂f_∂u(prob, prob.f, uu, p)
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

function NonlinearSolveBase.nonlinearsolve_forwarddiff_solve(
        prob::NonlinearLeastSquaresProblem, alg, args...; kwargs...)
    p = Utils.value(prob.p)
    newprob = remake(prob; p, u0 = Utils.value(prob.u0))
    sol = solve(newprob, alg, args...; kwargs...)
    uu = sol.u

    # First check for custom `vjp` then custom `Jacobian` and if nothing is provided use
    # nested autodiff as the last resort
    if SciMLBase.has_vjp(prob.f)
        if SciMLBase.isinplace(prob)
            vjp_fn = @closure (du, u, p) -> begin
                resid = Utils.safe_similar(du, length(sol.resid))
                prob.f(resid, u, p)
                prob.f.vjp(du, resid, u, p)
                du .*= 2
                return nothing
            end
        else
            vjp_fn = @closure (u, p) -> begin
                resid = prob.f(u, p)
                return reshape(2 .* prob.f.vjp(resid, u, p), size(u))
            end
        end
    elseif SciMLBase.has_jac(prob.f)
        if SciMLBase.isinplace(prob)
            vjp_fn = @closure (du, u, p) -> begin
                J = Utils.safe_similar(du, length(sol.resid), length(u))
                prob.f.jac(J, u, p)
                resid = Utils.safe_similar(du, length(sol.resid))
                prob.f(resid, u, p)
                mul!(reshape(du, 1, :), vec(resid)', J, 2, false)
                return nothing
            end
        else
            vjp_fn = @closure (u, p) -> begin
                return reshape(2 .* vec(prob.f(u, p))' * prob.f.jac(u, p), size(u))
            end
        end
    else
        # For small problems, nesting ForwardDiff is actually quite fast
        autodiff = length(uu) + length(sol.resid) ≥ 50 ?
                   NonlinearSolveBase.select_reverse_mode_autodiff(prob, nothing) :
                   AutoForwardDiff()

        if SciMLBase.isinplace(prob)
            vjp_fn = @closure (du, u, p) -> begin
                resid = Utils.safe_similar(du, length(sol.resid))
                prob.f(resid, u, p)
                # Using `Constant` lead to dual ordering issues
                ff = @closure (du, u) -> prob.f(du, u, p)
                resid2 = copy(resid)
                DI.pullback!(ff, resid2, (du,), autodiff, u, (resid,))
                @. du *= 2
                return nothing
            end
        else
            vjp_fn = @closure (u, p) -> begin
                v = prob.f(u, p)
                # Using `Constant` lead to dual ordering issues
                ff = Base.Fix2(prob.f, p)
                res = only(DI.pullback(ff, autodiff, u, (v,)))
                ArrayInterface.can_setindex(res) || return 2 .* res
                @. res *= 2
                return res
            end
        end
    end

    Jₚ = NonlinearSolveBase.nonlinearsolve_∂f_∂p(prob, vjp_fn, uu, newprob.p)
    Jᵤ = NonlinearSolveBase.nonlinearsolve_∂f_∂u(prob, vjp_fn, uu, newprob.p)
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
            @closure((du, u)->f(du, u, p)), Utils.safe_similar(u), u)
    end
    u isa Number && return ForwardDiff.derivative(Base.Fix2(f, p), u)
    return ForwardDiff.jacobian(Base.Fix2(f, p), u)
end

function NonlinearSolveBase.nonlinearsolve_dual_solution(u::Number, partials,
        ::Union{<:AbstractArray{<:Dual{T, V, P}}, Dual{T, V, P}}) where {T, V, P}
    return Dual{T, V, P}(u, partials)
end

function NonlinearSolveBase.nonlinearsolve_dual_solution(u::AbstractArray, partials,
        ::Union{<:AbstractArray{<:Dual{T, V, P}}, Dual{T, V, P}}) where {T, V, P}
    return map(((uᵢ, pᵢ),) -> Dual{T, V, P}(uᵢ, pᵢ), zip(u, Utils.restructure(u, partials)))
end

end
