module PolyalgSolutionTypeTests

using Test
using NonlinearSolveBase
using SciMLBase

# `build_solution_less_specialize` is the construction path used by
# `NonlinearSolvePolyAlgorithm`'s `solve!` (and the equivalent generated
# `__solve` path). Historically it used `Any` as the type slot for the
# `original` field so the returned `NonlinearSolution` type would be
# invariant to which sub-algorithm "won" the polyalg race.
#
# That `Any` field, however, leaves the solution type with mixed activity
# from Enzyme's perspective: Enzyme cannot tell whether `original` should be
# treated as active or constant, and bails out with `MixedReturnException`
# whenever the polyalg sits inside a reverse-mode differentiated function
# (#878). Pinning a single concrete sub-algorithm (e.g. `NewtonRaphson()`)
# avoids the polyalg entirely and works fine, which is the user-observed
# split between the two MWE variants in the issue.
#
# The fix drops the `original` payload on the polyalg's returned solution
# and types its slot as concrete `Nothing`. This keeps the
# no-specialization promise (no runtime-varying type to chase) while making
# the return type fully concrete for Enzyme. This test guards that
# invariant.
@testset "build_solution_less_specialize returns concrete `Nothing` original (#878)" begin
    f(u, p) = u .^ 2 .- p
    u0 = [1.0, 1.0]
    p = [2.0, 4.0]
    prob = NonlinearProblem{false}(f, u0, p)
    alg = nothing

    sol = NonlinearSolveBase.build_solution_less_specialize(
        prob, alg, u0, zeros(2); retcode = SciMLBase.ReturnCode.Success,
    )
    @test sol.original === nothing
    @test fieldtype(typeof(sol), :original) === Nothing

    # Even when a non-nothing `original` is passed by an internal caller, the
    # returned solution must still erase it. This keeps the type fully
    # concrete regardless of which sub-algorithm branch of the polyalg
    # produced it — which is the property the issue depends on. Polyalg
    # consumers should not have been relying on `.original` to inspect
    # internal sub-caches anyway (the field's documented purpose is to wrap
    # solutions from foreign solver ecosystems like NLsolve.jl, not internal
    # NonlinearSolve sub-caches).
    fake_inner = SciMLBase.build_solution(prob, alg, u0, zeros(2))
    sol2 = NonlinearSolveBase.build_solution_less_specialize(
        prob, alg, u0, zeros(2);
        retcode = SciMLBase.ReturnCode.Success, original = fake_inner,
    )
    @test sol2.original === nothing
    @test fieldtype(typeof(sol2), :original) === Nothing
    # Both invocations should produce the same solution type regardless of
    # whether `original` was supplied — this is what lets Enzyme's
    # `MixedReturnException` analysis pass.
    @test typeof(sol) === typeof(sol2)
end

end
