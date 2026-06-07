using BracketingNonlinearSolve

# Issue #860
a, b, c = 0.28299683548569865, 0.23110561490429102, -0.04293615418571393
f = (x, _) -> a * x^2 + b * x + c
tspan = (-18.26561304888476, -0.8216404439033063)

prob = IntervalNonlinearProblem{false}(f, tspan)
sol = solve(prob, ModAB(); abstol = 0.0, reltol = 0.0)

@test sol.retcode == ReturnCode.FloatingPointLimit
@test sol.left == -0.9726263117443217
@test sol.right == -0.9726263117443216
