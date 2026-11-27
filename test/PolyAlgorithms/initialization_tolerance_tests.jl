using NonlinearSolve, NonlinearSolveBase, SciMLBase

# The auxiliary initialization problem has a residual floor of 1e-10 (a stand-in for
# round-off in a stiff algebraic constraint): it cannot meet the default tolerance, but
# it does meet the `abstol = 1e-8` passed to `solve`. Initialization must use the
# tolerances given to `solve`, whichever path runs it: a solver cache, the one-shot
# polyalgorithm `solve` (which initializes on the problem before any cache exists), the
# bounds transform, or the multistart wrapper.
floor_residual(u, p) = u .- 0.75 .+ ifelse.(u .> 0.75, 1.0e-10, -1.0e-10)

function floor_initdata(; bounded = false)
    initprob = bounded ?
        NonlinearProblem(floor_residual, [0.5]; lb = 0.0, ub = 1.0) :
        NonlinearProblem(floor_residual, [0.5])
    return SciMLBase.OverrideInitData(initprob, nothing, sol -> sol.u, (_, sol) -> sol.u[1])
end

const TOL = (; abstol = 1.0e-8, reltol = 1.0e-8)

@testset "Initialization uses the solve tolerances" begin
    nf = NonlinearFunction((u, p) -> u .- p; initialization_data = floor_initdata())
    prob = NonlinearProblem(nf, [0.1], 0.0)

    # Control: the floor is above the default tolerance.
    @test solve(prob, FastShortcutNonlinearPolyalg()).retcode == ReturnCode.InitialFailure

    algs = (
        "default" => (), "FastShortcutNonlinearPolyalg" => (FastShortcutNonlinearPolyalg(),),
        "RobustMultiNewton" => (RobustMultiNewton(),), "NewtonRaphson" => (NewtonRaphson(),),
    )
    for (name, args) in algs
        @testset "$name, cached = $cached" for cached in (false, true)
            sol = cached ? solve!(init(prob, args...; TOL...)) : solve(prob, args...; TOL...)
            @test SciMLBase.successful_retcode(sol)
            @test sol.u ≈ [0.75]
        end
    end

    # A `SteadyStateProblem` is solved as the `NonlinearProblem` it converts to.
    ss = SteadyStateProblem(
        ODEFunction((u, p, t) -> p .- u; initialization_data = floor_initdata()), [0.1], 0.0
    )
    for alg in (FastShortcutNonlinearPolyalg(), RobustMultiNewton())
        sol = solve(ss, alg; TOL...)
        @test SciMLBase.successful_retcode(sol)
        @test sol.u ≈ [0.75]
    end
end

@testset "Bounded initialization uses the solve tolerances" begin
    nf = NonlinearFunction((u, p) -> u .- p; initialization_data = floor_initdata(bounded = true))
    prob = NonlinearProblem(nf, [0.1], 0.0; lb = 0.0, ub = 1.0)

    @test solve(prob, FastShortcutBoundedPolyalg()).retcode == ReturnCode.InitialFailure

    # Default (multistart over the bounded polyalgorithm) and the bounded polyalgorithm.
    # The bounds transform (an algorithm without native bounds support) also receives the
    # solve tolerances, but it does not report an initialization failure, so this
    # residual floor cannot tell the tolerances apart there.
    algs = ("default" => (), "FastShortcutBoundedPolyalg" => (FastShortcutBoundedPolyalg(),))
    for (name, args) in algs
        @testset "$name, cached = $cached" for cached in (false, true)
            sol = cached ? solve!(init(prob, args...; TOL...)) : solve(prob, args...; TOL...)
            @test SciMLBase.successful_retcode(sol)
            @test sol.u ≈ [0.75]
        end
    end
end

@testset "Explicit `nothing` tolerances defer to the initialization algorithm" begin
    nf = NonlinearFunction((u, p) -> u .- p; initialization_data = floor_initdata())
    prob = NonlinearProblem(nf, [0.1], 0.0)
    initializealg = SciMLBase.OverrideInit(; TOL...)
    for alg in (SobolMultistart(), SimpleNewtonRaphson())
        sol = solve!(init(prob, alg; abstol = nothing, reltol = nothing, initializealg))
        @test SciMLBase.successful_retcode(sol)
        @test sol.u ≈ [0.75]
    end
end
