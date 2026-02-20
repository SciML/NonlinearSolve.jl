module BracketingNonlinearSolveForwardDiffExt

using CommonSolve: CommonSolve
using ForwardDiff: ForwardDiff, Dual, Partials
using NonlinearSolveBase: nonlinearsolve_forwarddiff_solve, nonlinearsolve_dual_solution
using SciMLBase: SciMLBase, IntervalNonlinearProblem

using BracketingNonlinearSolve: Bisection, Brent, Alefeld, Falsi, ITP, Ridder, ModAB

const DualIntervalNonlinearProblem{
    T,
    V,
    P,
} = IntervalNonlinearProblem{
    uType, iip, <:Union{<:Dual{T, V, P}, <:AbstractArray{<:Dual{T, V, P}}},
} where {uType, iip}

for algT in (Bisection, Brent, Alefeld, Falsi, ITP, Ridder, ModAB)
    @eval function CommonSolve.solve(
            prob::DualIntervalNonlinearProblem{T, V, P}, alg::$(algT), args...;
            kwargs...
        ) where {T, V, P}
        sol, partials = nonlinearsolve_forwarddiff_solve(prob, alg, args...; kwargs...)
        dual_soln = nonlinearsolve_dual_solution(sol.u, partials, prob.p)
        return SciMLBase.build_solution(
            prob, alg, dual_soln, sol.resid;
            sol.retcode, sol.stats, sol.original,
            left = Dual{T, V, P}(sol.left, partials),
            right = Dual{T, V, P}(sol.right, partials)
        )
    end
end

# Handle the case where Duals are in both tspan AND p. This disambiguates the two
# methods below when both type aliases match. We forward to the p-Dual path since
# nonlinearsolve_forwarddiff_solve handles stripping Duals from both p and tspan.
const DualBothIntervalNonlinearProblem{
    T,
    V,
    P,
} = IntervalNonlinearProblem{
    iip, <:Tuple{Dual{T, V, P}, Dual{T, V, P}},
    <:Union{<:Dual{T, V, P}, <:AbstractArray{<:Dual{T, V, P}}},
} where {iip}

for algT in (Bisection, Brent, Alefeld, Falsi, ITP, Ridder, ModAB)
    @eval function CommonSolve.solve(
            prob::DualBothIntervalNonlinearProblem{T, V, P}, alg::$(algT), args...;
            kwargs...
        ) where {T, V, P}
        sol, partials = nonlinearsolve_forwarddiff_solve(prob, alg, args...; kwargs...)
        dual_soln = nonlinearsolve_dual_solution(sol.u, partials, prob.p)
        return SciMLBase.build_solution(
            prob, alg, dual_soln, sol.resid;
            sol.retcode, sol.stats, sol.original,
            left = Dual{T, V, P}(sol.left, partials),
            right = Dual{T, V, P}(sol.right, partials)
        )
    end
end

# Handle the case where Duals are in tspan (bracket endpoints) rather than in p.
# This occurs in DiffEqBase callback root-finding where the integrator's time values
# are Dual numbers carrying parameter sensitivities, but p is not passed to the
# IntervalNonlinearProblem.
#
# The derivative of the root w.r.t. the boundary point is zero (the root is where
# f=0, independent of bracket position). However, f's closure may capture
# Dual-valued state (e.g. from an ODE integrator), making this a mixed case where
# dt*/dθ = -(∂f/∂θ)/(∂f/∂t) is nonzero via the implicit function theorem.
# We use finite differences for ∂f/∂t to avoid nested Dual tag conflicts.

const DualTspanIntervalNonlinearProblem{
    T,
    V,
    P,
} = IntervalNonlinearProblem{
    iip, <:Tuple{Dual{T, V, P}, Dual{T, V, P}},
} where {iip}

for algT in (Bisection, Brent, Alefeld, Falsi, ITP, Ridder, ModAB)
    @eval function CommonSolve.solve(
            prob::DualTspanIntervalNonlinearProblem{T, V, P}, alg::$(algT), args...;
            kwargs...
        ) where {T, V, P}
        # Strip Duals from tspan and wrap f to strip any Dual output
        tspan = (ForwardDiff.value(prob.tspan[1]), ForwardDiff.value(prob.tspan[2]))
        f_orig = prob.f
        f_stripped(t, p) = ForwardDiff.value(f_orig(t, p))
        newprob = IntervalNonlinearProblem{false}(f_stripped, tspan, prob.p; prob.kwargs...)
        sol = CommonSolve.solve(newprob, alg, args...; kwargs...)

        # Check if f returns Duals (mixed case: closure captures Dual-valued state)
        root = sol.u
        f_at_root = prob.f(root, prob.p)

        if f_at_root isa Dual
            # Implicit function theorem: dt*/dθ = -(∂f/∂θ) / (∂f/∂t)
            # ∂f/∂t via central finite differences (avoids nested Dual tags)
            h = cbrt(eps(typeof(root))) * max(one(root), abs(root))
            dfdt = (f_stripped(root + h, prob.p) - f_stripped(root - h, prob.p)) / (2 * h)
            dfdθ = ForwardDiff.partials(f_at_root)
            partials = -dfdθ / dfdt
        else
            # Pure tspan-Dual case: derivative w.r.t. boundary is zero
            partials = Partials{P, V}(ntuple(_ -> zero(V), Val(P)))
        end

        dual_u = Dual{T, V, P}(sol.u, partials)
        left = Dual{T, V, P}(sol.left, partials)
        right = Dual{T, V, P}(sol.right, partials)

        return SciMLBase.build_solution(
            prob, alg, dual_u, sol.resid;
            sol.retcode, sol.stats, sol.original,
            left = left, right = right
        )
    end
end

end
