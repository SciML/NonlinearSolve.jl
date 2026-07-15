import CommonSolve
using LinearAlgebra, NonlinearSolveFirstOrder, SciMLBase

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
    @test default_policy.max_residual_ratio == 1
    @test NewtonRaphson().jacobian_reuse === nothing
    @test NewtonRaphson(jacobian_reuse = false).jacobian_reuse === nothing
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
        prob, NewtonRaphson(jacobian_reuse = true);
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
        prob, NewtonRaphson(jacobian_reuse = JacobianReuse(max_age = 2));
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
        divergent_prob, NewtonRaphson(jacobian_reuse = true);
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
            linesearch = FailAfterFirstLineSearch(), jacobian_reuse = true
        );
        abstol = 1.0e-14,
        reltol = 1.0e-14,
        verbose = false
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

@testset "manual override and reinit" begin
    jacobian_calls = Ref(0)
    prob = counted_problem(jacobian_calls)
    cache = init(
        prob, NewtonRaphson(jacobian_reuse = true);
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
        prob, TrustRegion(jacobian_reuse = true);
        abstol = 1.0e-10, reltol = 1.0e-10
    )

    @test SciMLBase.successful_retcode(sol)
    @test maximum(abs, sol.resid) ≤ 1.0e-10
    @test jacobian_calls[] < sol.stats.nsteps

    rejection_prob = NonlinearProblem(
        NonlinearFunction((u, p) -> u^3 - p; jac = (u, p) -> 3u^2), 0.5, 2.0
    )
    rejection_cache = init(
        rejection_prob, TrustRegion(jacobian_reuse = true);
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
            TrustRegion(step_threshold = 2, jacobian_reuse = true),
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
