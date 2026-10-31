using NonlinearSolveFirstOrder
using NonlinearSolveBase: NonlinearSolveBase, AbsNormTerminationMode, get_u, get_fu,
    not_terminated, refresh_residual!, supports_deferred_residual
using LineSearch: BackTracking

const fcalls = Ref(0)
f(u, p) = (fcalls[] += 1; u^3)
# Analytic, so `f` is only ever called to evaluate the residual.
df(u, p) = 3u^2
prob = NonlinearProblem(NonlinearFunction(f; jac = df), 1.0)

absnorm() = AbsNormTerminationMode(Base.Fix1(maximum, abs))

@testset "deferral is offered only where it is unobservable" begin
    @test supports_deferred_residual(
        init(prob, NewtonRaphson(); termination_condition = absnorm())
    )
    refusing = [
        # The default is `AbsNormSafeBestTerminationMode(...; max_stalled_steps = 32)`,
        # whose stall test reads a displacement a deferred step reports as identically zero.
        init(prob, NewtonRaphson()),
        # A globalized step consumes the residual as it goes.
        init(prob, TrustRegion(); termination_condition = absnorm()),
        init(
            prob, NewtonRaphson(; linesearch = BackTracking());
            termination_condition = absnorm()
        ),
        # The trace records the residual at the iterate the step landed on.
        init(
            prob, NewtonRaphson();
            termination_condition = absnorm(), store_trace = Val(true)
        ),
        # A polyalgorithm keeps its state on the active branch and defers nothing.
        init(prob, RobustMultiNewton(); termination_condition = absnorm()),
    ]
    for cache in refusing
        @test !supports_deferred_residual(cache)
        # A driver syncs before every read of the residual without asking who it is talking
        # to, so a cache that never defers still has to answer, and for free.
        fcalls[] = 0
        @test refresh_residual!(cache) === nothing
        @test fcalls[] == 0
    end
end

@testset "a step whose deferral would be misread evaluates anyway" begin
    # `abstol = 0` is unreachable, so the solve runs to `maxiters` with the iterate moving
    # every step. Honouring `evaluate_residual = false` under the default termination mode
    # would leave `u_cache == u`, and its stall test would call that motionless-looking
    # iterate a stall on step 33.
    cache = init(prob, NewtonRaphson(); abstol = 0.0, maxiters = 40)
    while not_terminated(cache)
        step!(cache; evaluate_residual = false)
        refresh_residual!(cache)
    end
    @test cache.nsteps == 40
    @test cache.retcode != ReturnCode.Stalled
end

@testset "a deferred step skips one residual and refresh_residual! pays it" begin
    cache = init(prob, NewtonRaphson(); termination_condition = absnorm())
    fcalls[] = 0
    step!(cache; evaluate_residual = false)
    @test fcalls[] == 0
    refresh_residual!(cache)
    @test fcalls[] == 1
    refresh_residual!(cache)
    @test fcalls[] == 1
end

@testset "a step that does nothing defers nothing" begin
    cache = init(prob, NewtonRaphson(); termination_condition = absnorm())
    solve!(cache)
    @test !not_terminated(cache)
    fcalls[] = 0
    step!(cache; evaluate_residual = false)
    refresh_residual!(cache)
    @test fcalls[] == 0
end

@testset "deferral does not move the iterates" begin
    plain = init(prob, NewtonRaphson(); termination_condition = absnorm())
    deferred = init(prob, NewtonRaphson(); termination_condition = absnorm())
    for _ in 1:40
        step!(plain)
        step!(deferred; evaluate_residual = false)
        refresh_residual!(deferred)
        @test get_u(plain) == get_u(deferred)
        @test get_fu(plain) == get_fu(deferred)
        @test plain.retcode == deferred.retcode
    end
    @test !not_terminated(plain)
end
