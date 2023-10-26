using CUDA, NonlinearSolve, LinearSolve

CUDA.allowscalar(false)

A = cu(rand(4, 4))
u0 = cu(rand(4))
b = cu(rand(4))

function f(du, u, p)
    du .= A * u .+ b
end

prob = NonlinearProblem(f, u0)

# TrustRegion is broken
# LimitedMemoryBroyden will diverge!
for alg in (NewtonRaphson(), LevenbergMarquardt(; linsolve = QRFactorization()),
    PseudoTransient(; alpha_initial = 10.0f0), GeneralKlement(), GeneralBroyden(),
    LimitedMemoryBroyden())
    @test_nowarn sol = solve(prob, alg; abstol = 1.0f-8, reltol = 1.0f-8)
end

f(u, p) = A * u .+ b

prob = NonlinearProblem{false}(f, u0)

# TrustRegion is broken
# LimitedMemoryBroyden will diverge!
for alg in (NewtonRaphson(), LevenbergMarquardt(; linsolve = QRFactorization()),
    PseudoTransient(; alpha_initial = 10.0f0), GeneralKlement(), GeneralBroyden(),
    LimitedMemoryBroyden())
    @test_nowarn sol = solve(prob, alg; abstol = 1.0f-8, reltol = 1.0f-8)
end
