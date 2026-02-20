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
# are Dual numbers, but p is not passed to the IntervalNonlinearProblem.
#
# The derivative of the root w.r.t. the bracket position is zero (the root is where
# f=0, independent of bracket position). However, f's closure may capture Dual-valued
# state (e.g. from an ODE integrator differentiating w.r.t. parameters), making this
# a mixed case where dt*/dθ is nonzero via the implicit function theorem:
#   dt*/dθ = -(∂f/∂θ) / (∂f/∂t)
#
# We compute this by evaluating f at the root and a nearby offset point, then applying
# the secant formula in Dual arithmetic. The offset uses sqrt(ε) spacing so the secant
# slope is well-conditioned (unlike machine-precision brackets where rounding dominates).
# Using the same Dual tag T as the tspan avoids introducing a foreign tag that would
# conflict with the caller's type system (e.g. ODE interpolant buffers typed for T).

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
        # Strip Duals from tspan and f returns for the solver
        tspan = (ForwardDiff.value(prob.tspan[1]), ForwardDiff.value(prob.tspan[2]))
        f_orig = prob.f
        f_stripped(t, p) = ForwardDiff.value(f_orig(t, p))
        newprob = IntervalNonlinearProblem{false}(f_stripped, tspan, prob.p; prob.kwargs...)
        sol = CommonSolve.solve(newprob, alg, args...; kwargs...)

        # Check if f's closure captures Dual-valued state (mixed case).
        # Both calls use Float64 time inputs, so they pass through ODE interpolants
        # without type conflicts. The Dual return comes from the closure's state.
        f_at_root = f_orig(sol.u, prob.p)

        if f_at_root isa Dual{T, V, P}
            # Mixed case: apply implicit function theorem via Dual secant step.
            # Evaluate f at the root (where f ≈ 0) and a nearby offset point
            # to get an accurate secant slope for df/dt.
            delta = sqrt(eps(V)) * max(one(V), abs(sol.u))
            t_hi = ForwardDiff.value(prob.tspan[2])
            t_lo = ForwardDiff.value(prob.tspan[1])
            if sol.u + delta ≤ t_hi
                t_other = sol.u + delta
            else
                t_other = sol.u - delta
            end
            f_at_other = f_orig(t_other, prob.p)

            zero_p = Partials{P, V}(ntuple(_ -> zero(V), Val(P)))
            a = Dual{T, V, P}(sol.u, zero_p)
            b = Dual{T, V, P}(t_other, zero_p)
            root_dual = a - f_at_root * (b - a) / (f_at_other - f_at_root)
            partials = ForwardDiff.partials(root_dual)
        else
            # Pure tspan-Dual case: derivative w.r.t. boundary is zero.
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
