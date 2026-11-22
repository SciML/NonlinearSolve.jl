using NonlinearSolve, SciMLBase, SciMLSensitivity, Zygote, Test
using ChainRulesCore: ChainRulesCore, Tangent, ZeroTangent

# The `solve_up` reverse rule must forward the whole solution cotangent to the
# backpass. Symbolic-indexing pullbacks such as `sol[obs]` return structural
# cotangents `(u = ..., prob = (p = ...))`: `u` can be zero while `prob.p`
# carries the parameter cotangent. Stripping the tangent to `∂sol.u` silently
# dropped that channel, producing zero gradients for observed-only losses.
#
# Consuming the structural tangent also requires the SciMLSensitivity release
# carrying https://github.com/SciML/SciMLSensitivity.jl/pull/1670 — older releases
# index `Δ.u` directly and cannot handle `u === ZeroTangent()`.
if Base.pkgversion(SciMLSensitivity) > v"7.119.8"
    @testset "structural solution cotangent reaches the backpass" begin
        ff(u, p) = u .^ 2 .- p
        nlprob = NonlinearProblem{false}(ff, [1.0, 2.0], [3.0, 2.0])
        _, pb = ChainRulesCore.rrule(
            NonlinearSolveBase.solve_up, nlprob, nothing,
            nlprob.u0, nlprob.p, NewtonRaphson();
            originator = SciMLBase.ChainRulesOriginator()
        )
        # u* = sqrt(p): the state-only cotangent gives dp = [1/(2√3), 1/(2√2)];
        # a `prob.p` cotangent must accumulate on top of it.
        state_only = pb(Tangent{Any}(u = [1.0, 1.0]))[5]
        @test state_only ≈ [1 / (2 * sqrt(3.0)), 1 / (2 * sqrt(2.0))] atol = 1.0e-8
        param_only = pb(Tangent{Any}(u = ZeroTangent(), prob = (p = [0.5, 0.5],)))[5]
        @test param_only ≈ [0.5, 0.5] atol = 1.0e-8
        both = pb(Tangent{Any}(u = [1.0, 1.0], prob = (p = [0.5, 0.5],)))[5]
        @test both ≈ [0.5 + 1 / (2 * sqrt(3.0)), 0.5 + 1 / (2 * sqrt(2.0))] atol = 1.0e-8
    end
end
