using SimpleNonlinearSolve

ff(u, p) = u .* u .- p
u0 = ones(2, 3)
p = 2.0
vecprob = NonlinearProblem(ff, vec(u0), p)
prob = NonlinearProblem(ff, u0, p)

@testset "$(alg)" for alg in (SimpleKlement(), SimpleBroyden(), SimpleNewtonRaphson(),
    SimpleDFSane(), SimpleLimitedMemoryBroyden(; threshold = Val(2)), SimpleTrustRegion())
    @test vec(solve(prob, alg).u) ≈ solve(vecprob, alg).u
end
