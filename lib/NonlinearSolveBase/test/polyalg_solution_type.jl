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
# That `Any` field, however, left the solution type with mixed activity from
# Enzyme's perspective: Enzyme could not tell whether `original` should be
# treated as active or constant, and bailed out with `MixedReturnException`
# whenever the polyalg sat inside a reverse-mode differentiated function
# (#878). Pinning a single concrete sub-algorithm (e.g. `NewtonRaphson()`)
# avoided the polyalg entirely and worked fine, which was the user-observed
# split between the two MWE variants in the issue.
#
# The fix carries a `store_original::Val` flag on the polyalg (default
# `Val(false)`, matching `SCCAlg`'s precedent). When `Val(false)` the
# returned solution's `original` field is concrete `Nothing`, fully
# Enzyme-friendly. When `Val(true)` the legacy `Any` payload is restored for
# users who want to inspect the winning sub-solution.
@testset "build_solution_less_specialize default (Val(false)) drops `original` (#878)" begin
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
    # default-branch result must still erase it. This keeps the type fully
    # concrete regardless of which sub-algorithm branch of the polyalg
    # produced it — which is the property the issue depends on.
    fake_inner = SciMLBase.build_solution(prob, alg, u0, zeros(2))
    sol2 = NonlinearSolveBase.build_solution_less_specialize(
        prob, alg, u0, zeros(2);
        retcode = SciMLBase.ReturnCode.Success, original = fake_inner,
    )
    @test sol2.original === nothing
    @test fieldtype(typeof(sol2), :original) === Nothing
    @test typeof(sol) === typeof(sol2)
end

@testset "build_solution_less_specialize opt-in (Val(true)) keeps `original`" begin
    f(u, p) = u .^ 2 .- p
    u0 = [1.0, 1.0]
    p = [2.0, 4.0]
    prob = NonlinearProblem{false}(f, u0, p)
    alg = nothing

    fake_inner = SciMLBase.build_solution(prob, alg, u0, zeros(2))
    sol = NonlinearSolveBase.build_solution_less_specialize(
        prob, alg, u0, zeros(2);
        retcode = SciMLBase.ReturnCode.Success, original = fake_inner,
        store_original = Val(true),
    )
    @test sol.original === fake_inner
    # Legacy `Any` slot — required because the polyalg branch chosen at
    # runtime varies the concrete type of the payload.
    @test fieldtype(typeof(sol), :original) === Any
end

@testset "NonlinearSolvePolyAlgorithm carries store_original (default Val(false))" begin
    alg_default = NonlinearSolveBase.NonlinearSolvePolyAlgorithm((nothing,))
    @test alg_default.store_original === Val(false)

    alg_opt = NonlinearSolveBase.NonlinearSolvePolyAlgorithm(
        (nothing,); store_original = Val(true)
    )
    @test alg_opt.store_original === Val(true)
end

end
