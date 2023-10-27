using LinearAlgebra, NonlinearSolve, StaticArrays, Test

# this is infeasible
function f1!(out, u, p)
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
function f1(u, p)
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
    return [a - 42.0e6, e - 1e-5, i - 1e-5]
end

@testset "[IIP] Infeasible" begin
    u0 = [0.0, 0.0, 0.0]
    prob = NonlinearProblem(f1!, u0)
    sol = solve(prob)

    @test all(!isnan, sol.u)
    @test !SciMLBase.successful_retcode(sol.retcode)
end

@testset "[OOP] Infeasible" begin
    u0 = [0.0, 0.0, 0.0]
    prob = NonlinearProblem(f1, u0)
    sol = solve(prob)

    @test all(!isnan, sol.u)
    @test !SciMLBase.successful_retcode(sol.retcode)

    try
        u0 = @SVector [0.0, 0.0, 0.0]
        prob = NonlinearProblem(f1, u0)
        sol = solve(prob)

        @test all(!isnan, sol.u)
        @test !SciMLBase.successful_retcode(sol.retcode)
    catch err
        # Static Arrays has different default linearsolve which throws an error
        @test err isa SingularException
    end
end
