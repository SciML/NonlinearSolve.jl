using NonlinearSolve

function f(du, u, p)
    s1, s1s2, s2 = u
    k1, c1, Δt = p

    du[1] = -0.25 * c1 * k1 * s1 * s2
    du[2] = 0.25 * c1 * k1 * s1 * s2
    return du[3] = -0.25 * c1 * k1 * s1 * s2
end

prob = NonlinearProblem(f, [2.0, 2.0, 2.0], [1.0, 2.0, 2.5])
sol = solve(prob; abstol = 1.0e-9)
@test SciMLBase.successful_retcode(sol)
