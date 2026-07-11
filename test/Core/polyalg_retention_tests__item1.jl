# Best-subalgorithm retention on the NonlinearSolvePolyAlgorithm cache
# (`reinit!(cache; retain_best = true)`): sticky start on the last winner, lazy
# subcache reinitialization, upward escalation, wrap-around floored at the
# algorithm's `start_index`, and exact status-quo behavior when retention is off.
using NonlinearSolve, SciMLBase

using Test

# Residual-call counter shared by every subalgorithm of the polyalgorithm (AD dual
# evaluations count too, which is fine: the assertions below are either exact reinit!
# counts or generous bounds).
const NF = Ref(0)
fcubic(u, p) = (NF[] += 1; u .^ 3 .- 2.0)
const ROOT = cbrt(2.0)

# Subalgorithm behaviors this file relies on (verified standalone): NewtonRaphson
# succeeds from u0 = 100 but stalls from u0 = 0 (singular Jacobian at the origin);
# Broyden (identity-initialized, derivative-free) succeeds from 0, 1.2, and 100.

@testset "retention picks the winning subalgorithm on the next solve" begin
    alg = NonlinearSolvePolyAlgorithm((NewtonRaphson(), Broyden()))
    prob = NonlinearProblem(fcubic, [0.0])
    cache = init(prob, alg)
    sol1 = solve!(cache)
    @test SciMLBase.successful_retcode(sol1)
    @test cache.best == 2      # Newton stalled at the singular point; Broyden won
    newton_nsteps = cache.caches[1].nsteps
    @test newton_nsteps > 0    # Newton was attempted before Broyden

    # Lazy reinit: a retaining reinit! reinitializes ONLY the retained subcache
    NF[] = 0
    reinit!(cache, [1.2]; retain_best = true)
    @test cache.current == 2
    @test NF[] == 1            # one residual evaluation, not one per subalgorithm

    NF[] = 0
    sol2 = solve!(cache)
    @test SciMLBase.successful_retcode(sol2)
    @test sol2.u[1] ≈ ROOT atol = 1.0e-6
    @test cache.best == 2
    # Newton's subcache was neither reinitialized nor stepped again
    @test cache.caches[1].nsteps == newton_nsteps
    # warm Broyden alone; generous bound, far below a Newton-then-Broyden rerun
    @test NF[] ≤ 15
end

@testset "escalation continues up the ladder when the sticky rung fails" begin
    alg = NonlinearSolvePolyAlgorithm((NewtonRaphson(), Broyden()))
    prob = NonlinearProblem(fcubic, [100.0])
    cache = init(prob, alg)
    sol1 = solve!(cache)
    @test SciMLBase.successful_retcode(sol1)
    @test cache.best == 1                  # Newton wins from 100
    reinit!(cache, [0.0]; retain_best = true)
    @test cache.current == 1
    sol2 = solve!(cache)                   # Newton stalls at 0 → escalate to Broyden
    @test SciMLBase.successful_retcode(sol2)
    @test sol2.u[1] ≈ ROOT atol = 1.0e-6
    @test cache.best == 2
end

@testset "wrap-around reaches the skipped cheaper rungs" begin
    alg = NonlinearSolvePolyAlgorithm((Broyden(), NewtonRaphson()))
    prob = NonlinearProblem(fcubic, [0.0])
    cache = init(prob, alg)
    # simulate a previous solve won by the LAST rung, as a continuation driver
    # retaining Newton across steps would have it
    cache.best = 2
    reinit!(cache, [0.0]; retain_best = true)
    @test cache.current == 2
    sol = solve!(cache)      # Newton stalls at 0 → ladder exhausted → wrap to rung 1
    @test cache.wrapped
    @test SciMLBase.successful_retcode(sol)
    @test sol.u[1] ≈ ROOT atol = 1.0e-6
    @test cache.best == 1
end

@testset "wrap-around never goes below the algorithm's start_index" begin
    alg = NonlinearSolvePolyAlgorithm((Broyden(), NewtonRaphson()); start_index = 2)
    prob = NonlinearProblem(fcubic, [0.0])
    cache = init(prob, alg)
    cache.best = 2
    reinit!(cache, [0.0]; retain_best = true)
    @test cache.current == 2
    sol = solve!(cache)
    # Newton stalls and rung 1 is excluded by start_index: no wrap, failed solve
    @test !cache.wrapped
    @test !SciMLBase.successful_retcode(sol)
    @test cache.caches[1].nsteps == 0     # Broyden was never attempted
end

@testset "periodic re-probe rediscovers a cheaper subalgorithm" begin
    alg = NonlinearSolvePolyAlgorithm((Broyden(), NewtonRaphson()))
    prob = NonlinearProblem(fcubic, [1.2])
    cache = init(prob, alg)
    cache.best = 2     # as if Newton had rescued a transient Broyden failure earlier
    for k in 1:7
        reinit!(cache, [1.2]; retain_best = true)
        @test cache.current == 2         # sticky on the retained rung
    end
    reinit!(cache, [1.2]; retain_best = true)    # 8th retained reinit! re-probes
    @test cache.current == 1
    sol = solve!(cache)
    @test SciMLBase.successful_retcode(sol)
    @test cache.best == 1     # Broyden won the re-probe and is retained again
    reinit!(cache, [1.2]; retain_best = true)
    @test cache.current == 1
end

@testset "retention off is the status-quo full restart" begin
    alg = NonlinearSolvePolyAlgorithm((NewtonRaphson(), Broyden()))
    prob = NonlinearProblem(fcubic, [0.0])
    cache = init(prob, alg)
    sol1 = solve!(cache)
    @test SciMLBase.successful_retcode(sol1)
    @test cache.best == 2
    NF[] = 0
    reinit!(cache, [1.2])                  # plain reinit!: no retention
    @test cache.current == 1               # ladder restarts at start_index
    @test cache.retain_best == false
    @test NF[] == 2                        # every subcache reinitialized eagerly
    sol2 = solve!(cache)
    @test SciMLBase.successful_retcode(sol2)
    @test sol2.u[1] ≈ ROOT atol = 1.0e-6
end
