using CUDA, NonlinearSolve, LinearSolve

CUDA.allowscalar(false)

A = cu(rand(4, 4))
u0 = cu(rand(4))
b = cu(rand(4))

linear_f(du, u, p) = (du .= A * u .+ b)

prob = NonlinearProblem(linear_f, u0)

for alg in (NewtonRaphson(), LevenbergMarquardt(; linsolve = QRFactorization()),
    PseudoTransient(; alpha_initial = 1.0f0), GeneralKlement(), GeneralBroyden(),
    LimitedMemoryBroyden(), TrustRegion())
    @test_nowarn sol = solve(prob, alg; abstol = 1.0f-8, reltol = 1.0f-8)
end

linear_f(u, p) = A * u .+ b

prob = NonlinearProblem{false}(linear_f, u0)

for alg in (NewtonRaphson(), LevenbergMarquardt(; linsolve = QRFactorization()),
    PseudoTransient(; alpha_initial = 1.0f0), GeneralKlement(), GeneralBroyden(),
    LimitedMemoryBroyden(), TrustRegion())
    @test_nowarn sol = solve(prob, alg; abstol = 1.0f-8, reltol = 1.0f-8)
end
