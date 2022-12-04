using CUDA, NonlinearSolve

A = cu(rand(4, 4))
u0 = cu(rand(4))
b = cu(rand(4))

function f(du, u, p)
    du .= A * u .+ b
end

prob = NonlinearProblem(f, u0)
sol = solve(prob, NewtonRaphson())
