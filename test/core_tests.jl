@testitem "NLLS Analytic Jacobian" tags=[:core] begin
    dataIn = 1:10
    f(x, p) = x[1] * dataIn .^ 2 .+ x[2] * dataIn .+ x[3]
    dataOut = f([1, 2, 3], nothing) + 0.1 * randn(10, 1)

    resid(x, p) = f(x, p) - dataOut
    jac(x, p) = [dataIn .^ 2 dataIn ones(10, 1)]
    x0 = [1, 1, 1]

    prob = NonlinearLeastSquaresProblem(resid, x0)
    sol1 = solve(prob)

    nlfunc = NonlinearFunction(resid; jac)
    prob = NonlinearLeastSquaresProblem(nlfunc, x0)
    sol2 = solve(prob)

    @test sol1.u ≈ sol2.u
end

@testitem "Basic PolyAlgorithms" tags=[:nopre] begin
    f(u, p) = u .* u .- 2
    u0 = [1.0, 1.0]

    prob = NonlinearProblem(f, u0)

    polyalgs = (
        RobustMultiNewton(), FastShortcutNonlinearPolyalg(), nothing, missing,
        NonlinearSolvePolyAlgorithm((Broyden(), LimitedMemoryBroyden()))
    )

    @testset "Direct Solve" begin
        @testset for alg in polyalgs
            alg = alg === missing ? () : (alg,)
            sol = solve(prob, alg...; abstol = 1e-9)
            @test SciMLBase.successful_retcode(sol)
            err = maximum(abs, f(sol.u, 2.0))
            @test err < 1e-9
        end
    end

    @testset "Caching Interface" begin
        @testset for alg in polyalgs
            alg = alg === missing ? () : (alg,)
            cache = init(prob, alg...; abstol = 1e-9)
            solver = solve!(cache)
            @test SciMLBase.successful_retcode(solver)
        end
    end

    @testset "Step Interface" begin
        @testset for alg in polyalgs
            alg = alg === missing ? () : (alg,)
            cache = init(prob, alg...; abstol = 1e-9)
            for i in 1:10000
                step!(cache)
                cache.force_stop && break
            end
            @test SciMLBase.successful_retcode(cache.retcode)
        end
    end
end

@testitem "PolyAlgorithms Autodiff" tags=[:nopre] begin
    cache = zeros(2)
    function f(du, u, p)
        cache .= u .* u
        du .= cache .- 2
    end
    u0 = [1.0, 1.0]
    probN = NonlinearProblem{true}(f, u0)

    custom_polyalg = NonlinearSolvePolyAlgorithm((
        Broyden(; autodiff = AutoFiniteDiff()), LimitedMemoryBroyden())
    )

    # Uses the `__solve` function
    @test_throws MethodError solve(probN; abstol = 1e-9)
    @test_throws MethodError solve(probN, RobustMultiNewton())

    sol = solve(probN, RobustMultiNewton(; autodiff = AutoFiniteDiff()))
    @test SciMLBase.successful_retcode(sol)

    sol = solve(
        probN, FastShortcutNonlinearPolyalg(; autodiff = AutoFiniteDiff()); abstol = 1e-9
    )
    @test SciMLBase.successful_retcode(sol)

    quadratic_f(u::Float64, p) = u^2 - p

    prob = NonlinearProblem(quadratic_f, 2.0, 4.0)

    @test_throws MethodError solve(prob)
    @test_throws MethodError solve(prob, RobustMultiNewton())

    sol = solve(prob, RobustMultiNewton(; autodiff = AutoFiniteDiff()))
    @test SciMLBase.successful_retcode(sol)
end

@testitem "PolyAlgorithm Aliasing" tags=[:core] begin
    using NonlinearProblemLibrary

    # Use a problem that the initial solvers cannot solve and cause the initial value to
    # diverge. If we don't alias correctly, all the subsequent algorithms will also fail.
    prob = NonlinearProblemLibrary.nlprob_23_testcases["Generalized Rosenbrock function"].prob
    u0 = copy(prob.u0)
    prob = remake(prob; u0 = copy(u0))

    # If aliasing is not handled properly this will diverge
    sol = solve(
        prob; abstol = 1e-6, alias_u0 = true,
        termination_condition = AbsNormTerminationMode(Base.Fix1(maximum, abs))
    )

    @test sol.u === prob.u0
    @test SciMLBase.successful_retcode(sol.retcode)

    prob = remake(prob; u0 = copy(u0))

    cache = init(
        prob; abstol = 1e-6, alias_u0 = true,
        termination_condition = AbsNormTerminationMode(Base.Fix1(maximum, abs))
    )
    sol = solve!(cache)

    @test sol.u === prob.u0
    @test SciMLBase.successful_retcode(sol.retcode)
end

@testitem "Ensemble Nonlinear Problems" tags=[:nopre] begin
    prob_func(prob, i, repeat) = remake(prob; u0 = prob.u0[:, i])

    prob_nls_oop = NonlinearProblem((u, p) -> u .* u .- p, rand(4, 128), 2.0)
    prob_nls_iip = NonlinearProblem((du, u, p) -> du .= u .* u .- p, rand(4, 128), 2.0)
    prob_nlls_oop = NonlinearLeastSquaresProblem((u, p) -> u .^ 2 .- p, rand(4, 128), 2.0)
    prob_nlls_iip = NonlinearLeastSquaresProblem(
        NonlinearFunction{true}((du, u, p) -> du .= u .^ 2 .- p; resid_prototype = rand(4)),
        rand(4, 128), 2.0
    )

    for prob in (prob_nls_oop, prob_nls_iip, prob_nlls_oop, prob_nlls_iip)
        ensembleprob = EnsembleProblem(prob; prob_func)

        for ensemblealg in (EnsembleThreads(), EnsembleSerial())
            sim = solve(ensembleprob, nothing, ensemblealg; trajectories = size(prob.u0, 2))
            @test all(SciMLBase.successful_retcode, sim.u)
        end
    end
end

@testitem "BigFloat Support" tags=[:core] begin
    using LinearAlgebra

    fn_iip = NonlinearFunction{true}((du, u, p) -> du .= u .* u .- p)
    fn_oop = NonlinearFunction{false}((u, p) -> u .* u .- p)

    u0 = BigFloat[1.0, 1.0, 1.0]
    prob_iip_bf = NonlinearProblem{true}(fn_iip, u0, BigFloat(2))
    prob_oop_bf = NonlinearProblem{false}(fn_oop, u0, BigFloat(2))

    for alg in (NewtonRaphson(), Broyden(), Klement(), DFSane(), TrustRegion())
        sol = solve(prob_oop_bf, alg)
        @test norm(sol.resid, Inf) < 1e-6
        @test SciMLBase.successful_retcode(sol.retcode)

        sol = solve(prob_iip_bf, alg)
        @test norm(sol.resid, Inf) < 1e-6
        @test SciMLBase.successful_retcode(sol.retcode)
    end
end

@testitem "Singular Exception: Issue #153" tags=[:core] begin
    function f(du, u, p)
        s1, s1s2, s2 = u
        k1, c1, Δt = p

        du[1] = -0.25 * c1 * k1 * s1 * s2
        du[2] = 0.25 * c1 * k1 * s1 * s2
        du[3] = -0.25 * c1 * k1 * s1 * s2
    end

    prob = NonlinearProblem(f, [2.0, 2.0, 2.0], [1.0, 2.0, 2.5])
    sol = solve(prob; abstol = 1e-9)
    @test SciMLBase.successful_retcode(sol)
end

@testitem "Simple Scalar Problem: Issue #187" tags=[:core] begin
    using NaNMath

    # https://github.com/SciML/NonlinearSolve.jl/issues/187
    # If we use a General Nonlinear Solver the solution might go out of the domain!
    ff_interval(u, p) = 0.5 / 1.5 * NaNMath.log.(u ./ (1.0 .- u)) .- 2.0 * u .+ 1.0

    uspan = (0.02, 0.1)
    prob = IntervalNonlinearProblem(ff_interval, uspan)
    sol = solve(prob; abstol = 1e-9)
    @test SciMLBase.successful_retcode(sol)

    u0 = 0.06
    p = 2.0
    prob = NonlinearProblem(ff_interval, u0, p)
    sol = solve(prob; abstol = 1e-9)
    @test SciMLBase.successful_retcode(sol)
end

# Shooting Problem: Taken from BoundaryValueDiffEq.jl
# Testing for Complex Valued Root Finding. For Complex valued inputs we drop some of the
# algorithms which dont support those.
@testitem "Complex Valued Problems: Single-Shooting" tags=[:core] begin
    using OrdinaryDiffEqTsit5

    function ode_func!(du, u, p, t)
        du[1] = u[2]
        du[2] = -u[1]
        return nothing
    end

    function objective_function!(resid, u0, p)
        odeprob = ODEProblem{true}(ode_func!, u0, (0.0, 100.0), p)
        sol = solve(
            odeprob, Tsit5(), abstol = 1e-9, reltol = 1e-9, verbose = true)
        resid[1] = sol(0.0)[1]
        resid[2] = sol(100.0)[1] - 1.0
        return nothing
    end

    prob = NonlinearProblem{true}(objective_function!, [0.0, 1.0] .+ 1im)
    sol = solve(prob; abstol = 1e-10)
    @test SciMLBase.successful_retcode(sol)
    # This test is not meant to return success but test that all the default solvers can handle
    # complex valued problems
    @test_nowarn solve(prob; abstol = 1e-19, maxiters = 10)
    @test_nowarn solve(
        prob, RobustMultiNewton(eltype(prob.u0)); abstol = 1e-19, maxiters = 10
    )
end

@testitem "No AD" tags=[:core] begin
    no_ad_fast = FastShortcutNonlinearPolyalg(autodiff = AutoFiniteDiff())
    no_ad_robust = RobustMultiNewton(autodiff = AutoFiniteDiff())
    no_ad_algs = Set([no_ad_fast, no_ad_robust, no_ad_fast.algs..., no_ad_robust.algs...])

    @testset "Inplace" begin
        f_iip = Base.Experimental.@opaque (du, u, p) -> du .= u .* u .- p
        u0 = [0.5]
        prob = NonlinearProblem(f_iip, u0, 1.0)
        for alg in no_ad_algs
            sol = solve(prob, alg)
            @test isapprox(only(sol.u), 1.0)
            @test SciMLBase.successful_retcode(sol.retcode)
        end
    end

    @testset "Out of Place" begin
        f_oop = Base.Experimental.@opaque (u, p) -> u .* u .- p
        u0 = [0.5]
        prob = NonlinearProblem{false}(f_oop, u0, 1.0)
        for alg in no_ad_algs
            sol = solve(prob, alg)
            @test isapprox(only(sol.u), 1.0)
            @test SciMLBase.successful_retcode(sol.retcode)
        end
    end
end

@testitem "Infeasible" tags=[:core] begin
    using LinearAlgebra, StaticArrays

    # this is infeasible
    function f1_infeasible!(out, u, p)
        μ = 3.986004415e14
        x = 7000.0e3
        y = -6.970561549987071e-9
        z = -3.784706123246018e-9
        v_x = 8.550491684548064e-12 + u[1]
        v_y = 6631.60076191005 + u[2]
        v_z = 3600.665431405663 + u[3]
        r = @SVector [x, y, z]
        v = @SVector [v_x, v_y, v_z]
        h = cross(r, v)
        ev = cross(v, h) / μ - r / norm(r)
        i = acos(h[3] / norm(h))
        e = norm(ev)
        a = 1 / (2 / norm(r) - (norm(v)^2 / μ))
        out .= [a - 42.0e6, e - 1e-5, i - 1e-5]
        return nothing
    end

    function f1_infeasible(u, p)
        μ = 3.986004415e14
        x = 7000.0e3
        y = -6.970561549987071e-9
        z = -3.784706123246018e-9
        v_x = 8.550491684548064e-12 + u[1]
        v_y = 6631.60076191005 + u[2]
        v_z = 3600.665431405663 + u[3]
        r = [x, y, z]
        v = [v_x, v_y, v_z]
        h = cross(r, v)
        ev = cross(v, h) / μ - r / norm(r)
        i = acos(h[3] / norm(h))
        e = norm(ev)
        a = 1 / (2 / norm(r) - (norm(v)^2 / μ))
        return [a - 42.0e6, e - 1e-5, i - 1e-5]
    end

    u0 = [0.0, 0.0, 0.0]
    prob = NonlinearProblem(f1_infeasible!, u0)
    sol = solve(prob)

    @test all(!isnan, sol.u)
    @test !SciMLBase.successful_retcode(sol.retcode)
    @inferred solve(prob)

    u0 = [0.0, 0.0, 0.0]
    prob = NonlinearProblem(f1_infeasible, u0)
    sol = solve(prob)

    @test all(!isnan, sol.u)
    @test !SciMLBase.successful_retcode(sol.retcode)
    @inferred solve(prob)

    u0 = @SVector [0.0, 0.0, 0.0]
    prob = NonlinearProblem(f1_infeasible, u0)

    sol = solve(prob)
    @test all(!isnan, sol.u)
    @test !SciMLBase.successful_retcode(sol.retcode)
end

@testitem "NoInit Caching" tags=[:core] begin
    using LinearAlgebra

    solvers = [
        SimpleNewtonRaphson(), SimpleTrustRegion(), SimpleDFSane()
    ]

    prob = NonlinearProblem((u, p) -> u .^ 2 .- p, [0.1, 0.3], 2.0)

    for alg in solvers
        cache = init(prob, alg)
        sol = solve!(cache)
        @test SciMLBase.successful_retcode(sol)
        @test norm(sol.resid, Inf) ≤ 1e-6

        reinit!(cache; p = 5.0)
        @test cache.prob.p == 5.0
        sol = solve!(cache)
        @test SciMLBase.successful_retcode(sol)
        @test norm(sol.resid, Inf) ≤ 1e-6
        @test norm(sol.u .^ 2 .- 5.0, Inf) ≤ 1e-6
    end
end

@testitem "Out-of-place Matrix Resizing" tags=[:nopre] begin
    using StableRNGs

    ff(u, p) = u .* u .- p
    u0 = rand(StableRNG(0), 2, 2)
    p = 2.0
    vecprob = NonlinearProblem(ff, vec(u0), p)
    prob = NonlinearProblem(ff, u0, p)

    for alg in (
        NewtonRaphson(), TrustRegion(), LevenbergMarquardt(),
        PseudoTransient(), RobustMultiNewton(), FastShortcutNonlinearPolyalg(),
        Broyden(), Klement(), LimitedMemoryBroyden(; threshold = 2)
    )
        @test vec(solve(prob, alg).u) == solve(vecprob, alg).u
    end
end

@testitem "Inplace Matrix Resizing" tags=[:nopre] begin
    using StableRNGs

    fiip(du, u, p) = (du .= u .* u .- p)
    u0 = rand(StableRNG(0), 2, 2)
    p = 2.0
    vecprob = NonlinearProblem(fiip, vec(u0), p)
    prob = NonlinearProblem(fiip, u0, p)

    for alg in (
        NewtonRaphson(), TrustRegion(), LevenbergMarquardt(),
        PseudoTransient(), RobustMultiNewton(), FastShortcutNonlinearPolyalg(),
        Broyden(), Klement(), LimitedMemoryBroyden(; threshold = 2)
    )
        @test vec(solve(prob, alg).u) == solve(vecprob, alg).u
    end
end

@testitem "Singular Systems -- Auto Linear Solve Switching" tags=[:core] begin
    using LinearAlgebra

    function f!(du, u, p)
        du[1] = 2u[1] - 2
        du[2] = (u[1] - 4u[2])^2 + 0.1
    end

    u0 = [0.0, 0.0] # Singular Jacobian at u0

    prob = NonlinearProblem(f!, u0)

    sol = solve(prob) # This doesn't have a root so let's just test the switching
    @test sol.u≈[1.0, 0.25] atol=1e-3 rtol=1e-3

    function nlls!(du, u, p)
        du[1] = 2u[1] - 2
        du[2] = (u[1] - 4u[2])^2 + 0.1
        du[3] = 0
    end

    u0 = [0.0, 0.0]

    prob = NonlinearProblem(NonlinearFunction(nlls!, resid_prototype = zeros(3)), u0)

    solve(prob)
    @test sol.u≈[1.0, 0.25] atol=1e-3 rtol=1e-3
end

@testitem "No PolyesterForwardDiff for SArray" tags=[:core] begin
    using StaticArrays, PolyesterForwardDiff

    f_oop(u, p) = u .* u .- p

    N = 4
    u0 = SVector{N, Float64}(ones(N) .+ randn(N) * 0.01)

    nlprob = NonlinearProblem(f_oop, u0, 2.0)

    @test !(solve(nlprob, NewtonRaphson()).alg.autodiff isa AutoPolyesterForwardDiff)
end

@testitem "NonlinearLeastSquares ReturnCode" tags=[:core] begin
    f(u, p) = [1.0]
    nlf = NonlinearFunction(f; resid_prototype = zeros(1))
    prob = NonlinearLeastSquaresProblem(nlf, [1.0])
    sol = solve(prob)
    @test SciMLBase.successful_retcode(sol)
    @test sol.retcode == ReturnCode.StalledSuccess
end

@testitem "Default Algorithm Singular Handling" tags=[:nopre] begin
    f(u, p) = [u[1]^2 - 2u[1] + 1, sum(u)]
    prob = NonlinearProblem(f, [1.0, 1.0])
    sol = solve(prob)
    @test SciMLBase.successful_retcode(sol)
end

@testitem "NonNumberEltype error" tags=[:core] begin
    u0_broken = [rand(2), rand(2)]
    f(u,p) = u
    prob = NonlinearProblem(f, u0_broken)
    @test_throws SciMLBase.NonNumberEltypeError solve(prob)=
end
