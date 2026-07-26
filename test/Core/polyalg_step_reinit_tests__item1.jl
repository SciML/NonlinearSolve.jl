using NonlinearSolve
using SciMLBase
import NonlinearSolveBase
import SymbolicIndexingInterface as SII

# Regression: a polyalgorithm cache driven one `step!` at a time (as ODE/DAE implicit
# solvers do, rather than through `solve!`) must keep working after a warm-started
# `reinit!`. `NonlinearSolvePolyAlgorithmCache`'s `reinit!` reset `current`/`nsteps`/
# `stats` but not `force_stop`/`retcode`, so once a branch converged the guard
# `not_terminated(cache) = !force_stop && ...` stayed false and every later `step!`
# returned immediately, silently reusing the stale iterate. `solve!` drives the
# subcaches directly and so never hit this, which is why it went unnoticed.
#
# Also checks the companion gap: `get_u`/`get_fu` on a polyalgorithm cache (its `u`/`fu`
# live on the active subcache, so the generic `cache.u`/`cache.fu` fallbacks threw).

step_drive!(cache) = while NonlinearSolveBase.not_terminated(cache)
    SciMLBase.step!(cache)
end

f(u, p) = u .^ 2 .- p

@testset "step!-driven polyalgorithm cache survives reinit! ($(nameof(typeof(alg))))" for alg in (
    RobustMultiNewton(), FastShortcutNonlinearPolyalg(),
)
    prob = NonlinearProblem(f, [1.0], 4.0)
    cache = init(prob, alg)

    step_drive!(cache)
    @test SciMLBase.successful_retcode(cache.retcode)
    @test NonlinearSolveBase.get_u(cache) ≈ [2.0] atol = 1.0e-6

    # warm start at a new parameter: the root moves to sqrt(9) = 3. Without the
    # force_stop/retcode reset this step! loop no-ops and `u` stays at the reinit u0.
    SciMLBase.reinit!(cache, [1.0]; p = 9.0)
    @test NonlinearSolveBase.not_terminated(cache)
    step_drive!(cache)
    @test SciMLBase.successful_retcode(cache.retcode)
    @test NonlinearSolveBase.get_u(cache) ≈ [3.0] atol = 1.0e-6
    @test NonlinearSolveBase.get_fu(cache) ≈ [0.0] atol = 1.0e-6

    # get_u must agree with the SymbolicIndexingInterface state accessor
    @test NonlinearSolveBase.get_u(cache) == SII.state_values(cache)
end
