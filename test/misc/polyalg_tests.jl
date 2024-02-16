@testitem "Basic PolyAlgorithms" begin
    f(u, p) = u .* u .- 2
    u0 = [1.0, 1.0]
    probN = NonlinearProblem{false}(f, u0)

    custom_polyalg = NonlinearSolvePolyAlgorithm((Broyden(), LimitedMemoryBroyden()))

    # Uses the `__solve` function
    solver = solve(probN; abstol = 1e-9)
    @test SciMLBase.successful_retcode(solver)
    solver = solve(probN, RobustMultiNewton(); abstol = 1e-9)
    @test SciMLBase.successful_retcode(solver)
    solver = solve(probN, FastShortcutNonlinearPolyalg(); abstol = 1e-9)
    @test SciMLBase.successful_retcode(solver)
    solver = solve(probN, custom_polyalg; abstol = 1e-9)
    @test SciMLBase.successful_retcode(solver)

    # Test the caching interface
    cache = init(probN; abstol = 1e-9)
    solver = solve!(cache)
    @test SciMLBase.successful_retcode(solver)
    cache = init(probN, RobustMultiNewton(); abstol = 1e-9)
    solver = solve!(cache)
    @test SciMLBase.successful_retcode(solver)
    cache = init(probN, FastShortcutNonlinearPolyalg(); abstol = 1e-9)
    solver = solve!(cache)
    @test SciMLBase.successful_retcode(solver)
    cache = init(probN, custom_polyalg; abstol = 1e-9)
    solver = solve!(cache)
    @test SciMLBase.successful_retcode(solver)

    # Test the step interface
    cache = init(probN; abstol = 1e-9)
    for i in 1:10000
        step!(cache)
        cache.force_stop && break
    end
    @test SciMLBase.successful_retcode(cache.retcode)
    cache = init(probN, RobustMultiNewton(); abstol = 1e-9)
    for i in 1:10000
        step!(cache)
        cache.force_stop && break
    end
    @test SciMLBase.successful_retcode(cache.retcode)
    cache = init(probN, FastShortcutNonlinearPolyalg(); abstol = 1e-9)
    for i in 1:10000
        step!(cache)
        cache.force_stop && break
    end
    @test SciMLBase.successful_retcode(cache.retcode)
    cache = init(probN, custom_polyalg; abstol = 1e-9)
    for i in 1:10000
        step!(cache)
        cache.force_stop && break
    end
end

@testitem "Testing #153 Singular Exception" begin
    # https://github.com/SciML/NonlinearSolve.jl/issues/153
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

@testitem "Simple Scalar Problem #187" begin
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
@testitem "Complex Valued Problems: Single-Shooting" begin
    using OrdinaryDiffEq

    function ode_func!(du, u, p, t)
        du[1] = u[2]
        du[2] = -u[1]
        return nothing
    end

    function objective_function!(resid, u0, p)
        odeprob = ODEProblem{true}(ode_func!, u0, (0.0, 100.0), p)
        sol = solve(odeprob, Tsit5(), abstol = 1e-9, reltol = 1e-9, verbose = false)
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
        prob, RobustMultiNewton(eltype(prob.u0)); abstol = 1e-19, maxiters = 10)
end

@testitem "No AD" begin
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

@testsetup module InfeasibleFunction
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

# this is unfeasible
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

export f1_infeasible!, f1_infeasible
end

@testitem "[IIP] Infeasible" setup=[InfeasibleFunction] begin
    u0 = [0.0, 0.0, 0.0]
    prob = NonlinearProblem(f1_infeasible!, u0)
    sol = solve(prob)

    @test all(!isnan, sol.u)
    @test !SciMLBase.successful_retcode(sol.retcode)
end

@testitem "[OOP] Infeasible" setup=[InfeasibleFunction] begin
    using LinearAlgebra, StaticArrays

    u0 = [0.0, 0.0, 0.0]
    prob = NonlinearProblem(f1_infeasible, u0)
    sol = solve(prob)

    @test all(!isnan, sol.u)
    @test !SciMLBase.successful_retcode(sol.retcode)

    u0 = @SVector [0.0, 0.0, 0.0]
    prob = NonlinearProblem(f1_infeasible, u0)

    try
        sol = solve(prob)
        @test all(!isnan, sol.u)
        @test !SciMLBase.successful_retcode(sol.retcode)
    catch err
        @test err isa LinearAlgebra.SingularException
    end
end
