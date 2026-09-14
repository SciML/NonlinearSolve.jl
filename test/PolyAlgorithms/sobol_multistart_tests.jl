using NonlinearSolve, SciMLBase, SparseArrays
using LinearSolve, LinearAlgebra

# Regression for https://github.com/SciML/NonlinearSolve.jl/issues/1147: the
# local bounded polyalgorithm converges to a constrained stationary point on an
# active bound, while an interior root exists.
@testset "SobolMultistart recovers a bounded stall" begin
    f!(u, p) = [cos(u[2]) + sin(u[1]) - 0.5, sin(u[2]) + cos(u[1]) - 0.3]
    prob = NonlinearProblem(
        NonlinearFunction{false}(f!), [0.3, 1.0], nothing;
        lb = [-100.0, 0.0], ub = [100.0, 10.0]
    )
    plain = solve(prob, FastShortcutBoundedPolyalg())
    @test plain.retcode === ReturnCode.Stalled
    @test norm(plain.resid) > 0.1

    default_sol = solve(prob)
    @test SciMLBase.successful_retcode(default_sol)
    @test norm(default_sol.resid) < 1.0e-8

    for alg in (
            SobolMultistart(), SobolMultistart(TrustRegionReflective()),
            SobolMultistart(BoundedTrustRegion()),
        )
        sol = solve(prob, alg)
        @test SciMLBase.successful_retcode(sol)
        @test norm(sol.resid) < 1.0e-8
        @test all(prob.lb .<= sol.u .<= prob.ub)
        @test sol.u ≈ [-0.24457517905731213, 5.548652801868546] atol = 1.0e-6
        @test sol.original isa SciMLBase.NonlinearSolution
    end

    cached = solve!(init(prob, SobolMultistart()))
    @test SciMLBase.successful_retcode(cached)
    @test norm(cached.resid) < 1.0e-8
end

@testset "SobolMultistart deterministic starts always include u0" begin
    f!(u, p) = [cos(u[2]) + sin(u[1]) - 0.5, sin(u[2]) + cos(u[1]) - 0.3]
    prob = NonlinearProblem(
        NonlinearFunction{false}(f!), [0.3, 1.0], nothing;
        lb = [-100.0, 0.0], ub = [100.0, 10.0]
    )
    # A single start reproduces the direct bounded solve, including its stall.
    direct = solve(prob, FastShortcutBoundedPolyalg())
    single = solve(prob, SobolMultistart(; nstarts = 1, restoration = false))
    @test single.retcode === direct.retcode
    @test single.u ≈ direct.u

    first = solve(prob, SobolMultistart(; nstarts = 8, early_exit = false))
    second = solve(prob, SobolMultistart(; nstarts = 8, early_exit = false))
    @test first.u == second.u
    @test first.retcode === second.retcode
end

@testset "SobolMultistart restoration accepts an in-box root" begin
    # The merit landscape has a constrained stationary point at (2, 0) on the
    # lower bound; the unbounded probe from it reaches the only root, which
    # lies inside the box.
    f(u, p) = [u[1] + 3u[2], 4 - u[1] - 2u[2] - u[2]^3]
    prob = NonlinearProblem(
        NonlinearFunction{false}(f), [0.0, 0.5], nothing;
        lb = [-10.0, 0.0], ub = [10.0, 10.0]
    )
    direct = solve(prob, FastShortcutBoundedPolyalg())
    @test direct.retcode === ReturnCode.Stalled

    recovered = solve(prob, SobolMultistart(; nstarts = 1))
    @test SciMLBase.successful_retcode(recovered)
    @test norm(recovered.resid) < 1.0e-8
    @test all(prob.lb .<= recovered.u .<= prob.ub)
    @test recovered.u ≈ [-5.388965709778324, 1.7963219032594415] atol = 1.0e-6

    norestore = solve(prob, SobolMultistart(; nstarts = 1, restoration = false))
    @test norestore.retcode === ReturnCode.Stalled
end

@testset "SobolMultistart restoration reports out-of-box roots honestly" begin
    f!(u, p) = [cos(u[2]) + sin(u[1]) - 0.5, sin(u[2]) + cos(u[1]) - 0.3]
    prob = NonlinearProblem(
        NonlinearFunction{false}(f!), [0.3, 1.0], nothing;
        lb = [-100.0, 0.0], ub = [100.0, 10.0]
    )
    sol = @test_logs (:warn, r"outside the bounds") match_mode = :any solve(
        prob, SobolMultistart(; nstarts = 1)
    )
    @test sol.retcode === ReturnCode.Stalled
    @test !SciMLBase.successful_retcode(sol)
end

@testset "SobolMultistart never masks failure" begin
    # No root exists inside or outside the box: u^2 - 2 has only the
    # infeasible root sqrt(2) > 1.
    prob = NonlinearProblem((u, p) -> u^2 - 2, 0.5, nothing; lb = 0.0, ub = 1.0)
    sol = @test_logs (:warn, r"outside the bounds") match_mode = :any solve(
        prob, SobolMultistart(; nstarts = 4)
    )
    @test sol.retcode === ReturnCode.Stalled
    @test sol.u ≈ 1.0

    g(u, p) = [u[1]^2 + u[2]^2 + 1.0, u[1] - u[2]]
    gprob = NonlinearProblem(
        NonlinearFunction{false}(g), [0.5, 0.5], nothing;
        lb = [0.0, 0.0], ub = [1.0, 1.0]
    )
    gsol = solve(gprob, SobolMultistart(; nstarts = 4))
    @test gsol.retcode === ReturnCode.Stalled
    @test !SciMLBase.successful_retcode(gsol)
end

@testset "SobolMultistart constructor validation" begin
    @test_throws ArgumentError SobolMultistart(; nstarts = 0)
    @test_throws ArgumentError SobolMultistart(; search_scale = 0)
    @test_throws ArgumentError SobolMultistart(; search_scale = -1)
    @test SciMLBase.allowsbounds(SobolMultistart())
end

@testset "SobolMultistart with mixed and infinite bounds" begin
    f!(u, p) = [cos(u[2]) + sin(u[1]) - 0.5, sin(u[2]) + cos(u[1]) - 0.3]
    prob = NonlinearProblem(
        NonlinearFunction{false}(f!), [0.3, 1.0], nothing;
        lb = [-Inf, 0.0], ub = [Inf, 10.0]
    )
    sol = solve(prob, SobolMultistart(; nstarts = 8))
    @test SciMLBase.successful_retcode(sol)
    @test norm(sol.resid) < 1.0e-8
    @test all(prob.lb .<= sol.u .<= prob.ub)

    unbounded = SciMLBase.remake(prob; lb = nothing, ub = nothing)
    usol = solve(unbounded, SobolMultistart(; nstarts = 8))
    @test SciMLBase.successful_retcode(usol)
    @test norm(usol.resid) < 1.0e-8
end

@testset "SobolMultistart fixed coordinates and NLLS" begin
    f(u, p) = [u[1] + u[2] - 3.0, u[1] - u[2] - 1.0]
    prob = NonlinearProblem(
        NonlinearFunction{false}(f), [0.5, 0.5], nothing;
        lb = [-10.0, 1.0], ub = [10.0, 1.0]
    )
    sol = solve(prob, SobolMultistart(; nstarts = 4))
    @test SciMLBase.successful_retcode(sol)
    @test sol.u ≈ [2.0, 1.0]

    nprob = NonlinearLeastSquaresProblem(
        (u, p) -> [u[1] - p, 0.5], [0.0], 2.0; lb = 0.0, ub = 1.0
    )
    nsol = solve(nprob, SobolMultistart(; nstarts = 4))
    @test SciMLBase.successful_retcode(nsol)
    @test nsol.u ≈ [1.0]
end

@testset "SobolMultistart sparse and operator Jacobians" begin
    n = 20
    A = spdiagm(-1 => fill(-0.1, n - 1), 0 => fill(2.0, n), 1 => fill(-0.1, n - 1))
    target = fill(0.75, n)
    f!(r, u, p) = mul!(r, A, u - p)
    jac!(J, u, p) = copyto!(J, A)
    jvp!(w, v, u, p) = mul!(w, A, v)
    vjp!(w, v, u, p) = mul!(w, transpose(A), v)
    for matrix_free in (false, true)
        nf = NonlinearFunction(f!; jac = jac!, jac_prototype = copy(A), jvp = jvp!, vjp = vjp!)
        prob = NonlinearLeastSquaresProblem(nf, fill(0.1, n), target; lb = 0.0, ub = 1.0)
        alg = SobolMultistart(
            FastShortcutBoundedPolyalg(linsolve = matrix_free ? KrylovJL_LSMR() : nothing);
            nstarts = 3
        )
        sol = solve(prob, alg; abstol = 1.0e-9)
        @test SciMLBase.successful_retcode(sol)
        @test sol.u ≈ target atol = 1.0e-7
    end
end
