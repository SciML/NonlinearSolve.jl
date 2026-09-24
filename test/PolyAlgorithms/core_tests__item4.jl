using NonlinearSolve

cache = zeros(2)
function f(du, u, p)
    cache .= u .* u
    return du .= cache .- 2
end
u0 = [1.0, 1.0]
probN = NonlinearProblem{true}(f, u0)

custom_polyalg = NonlinearSolvePolyAlgorithm(
    (
        Broyden(; autodiff = AutoFiniteDiff()), LimitedMemoryBroyden(),
    )
)

# Uses the `__solve` function
@test_throws MethodError solve(probN; abstol = 1.0e-9)
@test_throws MethodError solve(probN, RobustMultiNewton())

sol = solve(probN, RobustMultiNewton(; autodiff = AutoFiniteDiff()))
@test SciMLBase.successful_retcode(sol)

sol = solve(
    probN, FastShortcutNonlinearPolyalg(; autodiff = AutoFiniteDiff()); abstol = 1.0e-9
)
@test SciMLBase.successful_retcode(sol)

quadratic_f(u::Float64, p) = u^2 - p

prob = NonlinearProblem(quadratic_f, 2.0, 4.0)

@test_throws MethodError solve(prob)
@test_throws MethodError solve(prob, RobustMultiNewton())

sol = solve(prob, RobustMultiNewton(; autodiff = AutoFiniteDiff()))
@test SciMLBase.successful_retcode(sol)
