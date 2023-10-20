using NonlinearSolve, Test

ff(u, p) = u .* u .- p
u0 = rand(2,2)
p = 2.0
vecprob = NonlinearProblem(ff, vec(u0), p)
prob = NonlinearProblem(ff, u0, p)

for alg in (NewtonRaphson(), TrustRegion(), LevenbergMarquardt(), PseudoTransient(), RobustMultiNewton(), FastShortcutNonlinearPolyalg())
    @test vec(solve(prob, alg).u) == solve(vecprob, alg).u
end

fiip(du, u, p) = (du .= u .* u .- p)
u0 = rand(2,2)
p = 2.0
vecprob = NonlinearProblem(fiip, vec(u0), p)
prob = NonlinearProblem(fiip, u0, p)

for alg in (NewtonRaphson(), TrustRegion(), LevenbergMarquardt(), PseudoTransient(), RobustMultiNewton(), FastShortcutNonlinearPolyalg())
    @test vec(solve(prob, alg).u) == solve(vecprob, alg).u
end