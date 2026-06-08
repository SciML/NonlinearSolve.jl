using NonlinearSolve

using SciMLBase
using NonlinearSolveBase

# The quasi-Newton `autodiff`-access fix lives in NonlinearSolveBase, so only
# exercise the full solve when a NonlinearSolveBase new enough to contain it is
# actually loaded. On Julia < 1.11 the umbrella resolves the *registered*
# NonlinearSolveBase (which predates the fix) because `[sources]` path redirects
# are ignored there; the fix itself is unit-tested in NonlinearSolveBase's own
# suite, which always runs against the in-repo code. See SciML/NonlinearSolve.jl#955.
if pkgversion(NonlinearSolveBase) >= v"2.30.3"
    # `x^2 - 4x + 3` has roots at 1 and 3; bounds select which root is reachable.
    f(u, p) = u .^ 2 .- 4 .* u .+ 3

    # The default polyalgorithm tries quasi-Newton methods (which have no `autodiff`
    # field) as part of its sequence; the bounds transform must not error on them.
    for alg in (nothing, FastShortcutNonlinearPolyalg(), Broyden(), Klement(), NewtonRaphson())
        prob = NonlinearProblem(f, [1.5], nothing; lb = [0.0], ub = [2.0])
        sol = alg === nothing ? solve(prob) : solve(prob, alg)
        @test SciMLBase.successful_retcode(sol)
        @test sol.u[1] ≈ 1.0 atol = 1.0e-6
        @test 0.0 <= sol.u[1] <= 2.0
    end
end
