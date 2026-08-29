import CommonSolve
using LinearAlgebra, LinearSolve, NonlinearSolveFirstOrder, SciMLBase
using NonlinearSolveBase: AbsNormTerminationMode
using NonlinearSolveFirstOrder: JACOBIAN_REUSE_SIZE_CUTOFF, resolve_jacobian_reuse,
    reuses_jacobian

# The default policy only reuses while the residual contracts hard, which the quadratically
# converging problems below do only at the very end. Tests that assert a step-by-step reuse
# pattern ask for the permissive policy instead.
const REUSE_WHILE_IMPROVING = JacobianReuse(max_age = 10, max_residual_ratio = 1)

struct FailAfterFirstLineSearch end

mutable struct FailAfterFirstLineSearchCache
    calls::Int
end

function CommonSolve.init(
        prob::SciMLBase.AbstractNonlinearProblem, ::FailAfterFirstLineSearch, fu, u;
        kwargs...
    )
    return FailAfterFirstLineSearchCache(0)
end

function CommonSolve.solve!(cache::FailAfterFirstLineSearchCache, u, δu)
    cache.calls += 1
    return (;
        retcode = cache.calls == 1 ? ReturnCode.Success : ReturnCode.Failure,
        step_size = cache.calls == 1 ? 1.0 : 0.0,
    )
end

function counted_problem(jacobian_calls)
    f(u, p) = u .* u .- p
    function jac(u, p)
        jacobian_calls[] += 1
        return Matrix(Diagonal(2 .* u))
    end
    return NonlinearProblem(NonlinearFunction(f; jac), ones(2), 2.0)
end

@testset "configuration" begin
    default_policy = JacobianReuse()
    @test default_policy.max_age == 10
    @test 0 < default_policy.max_residual_ratio < 1
    @test reuses_jacobian(default_policy)
    @test !reuses_jacobian(JacobianReuse(max_age = 1))
    @test NewtonRaphson().jacobian_reuse === nothing
    @test NewtonRaphson(jacobian_reuse = false).jacobian_reuse == JacobianReuse(max_age = 1)
    @test NewtonRaphson(jacobian_reuse = true).jacobian_reuse isa JacobianReuse
    @test NewtonRaphson(jacobian_reuse = default_policy).jacobian_reuse === default_policy

    for alg in (
            TrustRegion(jacobian_reuse = true),
            GaussNewton(jacobian_reuse = true),
            LevenbergMarquardt(jacobian_reuse = true),
            PseudoTransient(jacobian_reuse = true),
        )
        @test alg.jacobian_reuse isa JacobianReuse
    end
    @test all(
        alg -> alg.jacobian_reuse isa JacobianReuse,
        RobustMultiNewton(jacobian_reuse = true).algs
    )
    @test all(
        alg -> alg.jacobian_reuse isa JacobianReuse,
        FastShortcutNLLSPolyalg(jacobian_reuse = true).algs
    )

    @test_throws ArgumentError JacobianReuse(max_age = 0)
    @test_throws ArgumentError JacobianReuse(max_residual_ratio = -1)
    @test_throws ArgumentError JacobianReuse(max_residual_ratio = NaN)
    @test_throws ArgumentError JacobianReuse(0, 1)
    @test_throws ArgumentError JacobianReuse(1, NaN)
    @test_throws ArgumentError NewtonRaphson(jacobian_reuse = :invalid)
end

@testset "Newton refresh policy" begin
    jacobian_calls = Ref(0)
    prob = counted_problem(jacobian_calls)

    exact_sol = solve(prob, NewtonRaphson(); abstol = 1.0e-10, reltol = 1.0e-10)
    exact_jacobian_calls = jacobian_calls[]
    jacobian_calls[] = 0
    reuse_sol = solve(
        prob, NewtonRaphson(jacobian_reuse = REUSE_WHILE_IMPROVING);
        abstol = 1.0e-10, reltol = 1.0e-10
    )

    @test SciMLBase.successful_retcode(exact_sol)
    @test SciMLBase.successful_retcode(reuse_sol)
    @test maximum(abs, reuse_sol.resid) ≤ 1.0e-10
    @test jacobian_calls[] < exact_jacobian_calls
    @test reuse_sol.stats.njacs < exact_sol.stats.njacs
    @test reuse_sol.stats.nfactors < exact_sol.stats.nfactors

    jacobian_calls[] = 0
    cache = init(
        prob,
        NewtonRaphson(
            jacobian_reuse = JacobianReuse(max_age = 2, max_residual_ratio = 1)
        );
        abstol = 1.0e-14, reltol = 1.0e-14
    )
    step!(cache)
    @test jacobian_calls[] == 1
    @test !cache.make_new_jacobian
    step!(cache)
    @test jacobian_calls[] == 1
    @test cache.make_new_jacobian
    step!(cache)
    @test jacobian_calls[] == 2

    jacobian_calls[] = 0
    progress_guard_cache = init(
        prob,
        NewtonRaphson(
            jacobian_reuse = JacobianReuse(max_age = 10, max_residual_ratio = 0)
        );
        abstol = 1.0e-14, reltol = 1.0e-14
    )
    step!(progress_guard_cache)
    @test progress_guard_cache.make_new_jacobian
    step!(progress_guard_cache)
    @test jacobian_calls[] == 2

    divergent_prob = NonlinearProblem(
        NonlinearFunction((u, p) -> u^3 - p; jac = (u, p) -> 3u^2), 0.5, 2.0
    )
    divergent_cache = init(
        divergent_prob, NewtonRaphson(jacobian_reuse = REUSE_WHILE_IMPROVING);
        abstol = 1.0e-14, reltol = 1.0e-14
    )
    initial_residual = abs(divergent_cache.fu)
    step!(divergent_cache)
    @test abs(divergent_cache.fu) > initial_residual
    @test divergent_cache.make_new_jacobian
end

@testset "Line search stale-Jacobian retry" begin
    jacobian_calls = Ref(0)
    prob = counted_problem(jacobian_calls)
    cache = init(
        prob,
        NewtonRaphson(
            linesearch = FailAfterFirstLineSearch(),
            jacobian_reuse = REUSE_WHILE_IMPROVING
        );
        abstol = 1.0e-14,
        reltol = 1.0e-14,
        verbose = false,
        # Retained in `cache.kwargs`, so the retry must not splat them into `step!`.
        alias_u0 = false
    )

    step!(cache)
    @test !cache.make_new_jacobian
    @test jacobian_calls[] == 1

    step!(cache)
    @test cache.linesearch_cache.calls == 3
    @test jacobian_calls[] == 2
    @test cache.retcode == ReturnCode.InternalLineSearchFailed
    @test cache.force_stop
end

@testset "deferred residual still drives the policy" begin
    jacobian_calls = Ref(0)
    prob = counted_problem(jacobian_calls)
    cache = init(
        prob,
        NewtonRaphson(
            jacobian_reuse = JacobianReuse(max_age = 2, max_residual_ratio = 1)
        );
        abstol = 1.0e-14, termination_condition = AbsNormTerminationMode(Base.Fix1(maximum, abs))
    )
    step!(cache; evaluate_residual = false)
    @test cache.fu_deferred
    @test jacobian_calls[] == 1
    step!(cache; evaluate_residual = false)
    @test jacobian_calls[] == 1
    step!(cache)
    @test jacobian_calls[] == 2
end

@testset "matrix-free Jacobians disable the policy" begin
    prob = NonlinearProblem((u, p) -> u .* u .- p, ones(2), 2.0)
    cache = init(
        prob, NewtonRaphson(linsolve = KrylovJL_GMRES(), jacobian_reuse = true);
        abstol = 1.0e-10
    )
    @test cache.jacobian_reuse_cache === nothing
    @test SciMLBase.successful_retcode(solve!(cache))
end

@testset "manual override and reinit" begin
    jacobian_calls = Ref(0)
    prob = counted_problem(jacobian_calls)
    cache = init(
        prob, NewtonRaphson(jacobian_reuse = REUSE_WHILE_IMPROVING);
        abstol = 1.0e-14, reltol = 1.0e-14
    )

    step!(cache)
    @test jacobian_calls[] == 1
    @test !cache.make_new_jacobian
    step!(cache; recompute_jacobian = true)
    @test jacobian_calls[] == 2
    step!(cache; recompute_jacobian = false)
    @test jacobian_calls[] == 2

    reinit!(cache, ones(2))
    @test cache.make_new_jacobian
    @test cache.stats.njacs == 0
    step!(cache)
    @test jacobian_calls[] == 3
    @test cache.stats.njacs == 1

    jacobian_calls[] = 0
    exact_cache = init(
        prob, NewtonRaphson(); abstol = 1.0e-14, reltol = 1.0e-14
    )
    step!(exact_cache)
    @test exact_cache.make_new_jacobian
    step!(exact_cache; recompute_jacobian = false)
    @test jacobian_calls[] == 1
    @test exact_cache.make_new_jacobian
end

@testset "TrustRegion reuse" begin
    jacobian_calls = Ref(0)
    prob = counted_problem(jacobian_calls)
    sol = solve(
        prob, TrustRegion(jacobian_reuse = REUSE_WHILE_IMPROVING);
        abstol = 1.0e-10, reltol = 1.0e-10
    )

    @test SciMLBase.successful_retcode(sol)
    @test maximum(abs, sol.resid) ≤ 1.0e-10
    @test jacobian_calls[] < sol.stats.nsteps

    rejection_prob = NonlinearProblem(
        NonlinearFunction((u, p) -> u^3 - p; jac = (u, p) -> 3u^2), 0.5, 2.0
    )
    rejection_cache = init(
        rejection_prob, TrustRegion(jacobian_reuse = REUSE_WHILE_IMPROVING);
        abstol = 1.0e-14, reltol = 1.0e-14
    )
    step!(rejection_cache)
    step!(rejection_cache)
    @test !rejection_cache.make_new_jacobian
    state_before_rejection = rejection_cache.u
    step!(rejection_cache)
    @test rejection_cache.u == state_before_rejection
    @test rejection_cache.make_new_jacobian
    @test rejection_cache.stats.njacs == 1
    step!(rejection_cache)
    @test rejection_cache.stats.njacs == 2

    for alg in (
            TrustRegion(step_threshold = 2),
            TrustRegion(step_threshold = 2, jacobian_reuse = REUSE_WHILE_IMPROVING),
        )
        repeated_rejection_cache = init(
            rejection_prob, alg; abstol = 1.0e-14, reltol = 1.0e-14
        )
        initial_state = repeated_rejection_cache.u
        step!(repeated_rejection_cache)
        step!(repeated_rejection_cache)
        @test repeated_rejection_cache.u == initial_state
        @test !repeated_rejection_cache.make_new_jacobian
        @test repeated_rejection_cache.stats.njacs == 1
    end
end

sized_problem(n) = NonlinearProblem((u, p) -> u .* u .- p, fill(1.5, n), 2.0)

@testset "max_age = 1 reproduces exact Newton" begin
    for n in (1, JACOBIAN_REUSE_SIZE_CUTOFF - 1, JACOBIAN_REUSE_SIZE_CUTOFF)
        prob = sized_problem(n)
        for alg in (NewtonRaphson, TrustRegion)
            off = solve(
                prob, alg(jacobian_reuse = false); abstol = 1.0e-10, reltol = 1.0e-10
            )
            aged = solve(
                prob, alg(jacobian_reuse = JacobianReuse(max_age = 1));
                abstol = 1.0e-10, reltol = 1.0e-10
            )
            @test off.stats.njacs == aged.stats.njacs
            @test off.stats.nfactors == aged.stats.nfactors
            @test off.stats.nsteps == aged.stats.nsteps
            @test off.u == aged.u
        end
    end
end

@testset "the policy is a value, not a type" begin
    # Spelling "reuse off" as `max_age = 1` is what lets the size-based default pick a
    # policy without splitting the solver cache into two specializations.
    small = sized_problem(JACOBIAN_REUSE_SIZE_CUTOFF - 1)
    large = sized_problem(JACOBIAN_REUSE_SIZE_CUTOFF)
    for alg in (NewtonRaphson, TrustRegion)
        @test typeof(alg(jacobian_reuse = false)) === typeof(alg(jacobian_reuse = true))
        @test typeof(init(small, alg(); abstol = 1.0e-10)) ===
            typeof(init(large, alg(); abstol = 1.0e-10))
        @test typeof(@inferred solve(small, alg(); abstol = 1.0e-10, reltol = 1.0e-10)) ===
            typeof(@inferred solve(large, alg(); abstol = 1.0e-10, reltol = 1.0e-10))
    end
end

@testset "size-based default" begin
    below = sized_problem(JACOBIAN_REUSE_SIZE_CUTOFF - 1)
    atcut = sized_problem(JACOBIAN_REUSE_SIZE_CUTOFF)

    @test !reuses_jacobian(resolve_jacobian_reuse(nothing, below.u0))
    @test reuses_jacobian(resolve_jacobian_reuse(nothing, atcut.u0))
    # An explicit policy ignores the size entirely.
    @test reuses_jacobian(resolve_jacobian_reuse(JacobianReuse(), below.u0))
    @test !reuses_jacobian(resolve_jacobian_reuse(JacobianReuse(max_age = 1), atcut.u0))

    for alg in (NewtonRaphson, TrustRegion, GaussNewton, PseudoTransient)
        cache = init(below, alg(); abstol = 1.0e-10, reltol = 1.0e-10)
        @test !reuses_jacobian(cache.jacobian_reuse_cache.policy)
        cache = init(atcut, alg(); abstol = 1.0e-10, reltol = 1.0e-10)
        @test reuses_jacobian(cache.jacobian_reuse_cache.policy)
    end
end

@testset "a pinned Jacobian is never retried" begin
    # The retry recovers from the policy's own decision to reuse. A caller that pinned the
    # Jacobian owns that decision, whether or not the policy would also have reused, so
    # driving `step!` by hand keeps its behavior now that the default turns reuse on.
    for policy in (false, REUSE_WHILE_IMPROVING)
        jacobian_calls = Ref(0)
        prob = counted_problem(jacobian_calls)
        cache = init(
            prob,
            NewtonRaphson(
                linesearch = FailAfterFirstLineSearch(), jacobian_reuse = policy
            );
            abstol = 1.0e-14, reltol = 1.0e-14, verbose = false
        )

        step!(cache; recompute_jacobian = true)
        @test jacobian_calls[] == 1
        step!(cache; recompute_jacobian = false)
        @test jacobian_calls[] == 1
        @test cache.retcode == ReturnCode.InternalLineSearchFailed
        @test cache.force_stop
    end
end
