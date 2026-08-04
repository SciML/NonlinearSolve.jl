using NonlinearSolve
using SciMLBase
using ADTypes
using Test

const NLB = NonlinearSolve.NonlinearSolveBase

# `FastShortcutHomotopyPolyalg` is the named default continuation polyalgorithm for a
# `HomotopyProblem`: a `HomotopyPolyAlgorithm` (fast sweep, then robust arclength) whose inner
# corrector is a `FastShortcutNonlinearPolyalg` carrying the requested autodiff. It is the
# homotopy analogue of `FastShortcutNonlinearPolyalg`.

# branch selection: the target (λ = 1) system m² - 1 has roots ±1; from guess m = -3 a plain
# Newton solve of that system lands on the unphysical -1, while continuation from the
# simplified 2(m - 1) (anchored at +1) reaches the physical +1. A solve that returns +1 must
# therefore have been continued, not target-λ solved.
Hbranch(u, p, λ) = [(1 - λ) * 2 * (u[1] - 1) + λ * (u[1]^2 - 1)]

@testset "FastShortcutHomotopyPolyalg wiring" begin
    alg = FastShortcutHomotopyPolyalg()
    @test alg isa NLB.HomotopyPolyAlgorithm
    @test length(alg.algs) == 2
    @test alg.algs[1] isa NLB.HomotopySweep
    @test alg.algs[2] isa NLB.ArcLengthContinuation
    # the inner corrector of each stage is the FastShortcut nonlinear polyalgorithm
    @test alg.algs[1].inner isa NonlinearSolvePolyAlgorithm
    @test alg.algs[2].inner isa NonlinearSolvePolyAlgorithm
end

@testset "FastShortcutHomotopyPolyalg continues to the physical branch" begin
    prob = HomotopyProblem(Hbranch, [-3.0])
    sol = solve(prob, FastShortcutHomotopyPolyalg())
    @test SciMLBase.successful_retcode(sol)
    @test sol.u[1] ≈ 1.0 atol = 1.0e-6
    # a plain nonlinear algorithm solves only the target (λ = 1) system -> unphysical -1
    @test solve(prob, NewtonRaphson()).u[1] ≈ -1.0 atol = 1.0e-6
    # the finite-diff configuration is accepted and solves end-to-end
    sol_fd = solve(prob, FastShortcutHomotopyPolyalg(; autodiff = AutoFiniteDiff()))
    @test sol_fd.u[1] ≈ 1.0 atol = 1.0e-6
end

@testset "initialization_alg(::HomotopyProblem) routes to continuation" begin
    prob = HomotopyProblem(Hbranch, [-3.0])
    # the AbstractNonlinearProblem fallback would return a FastShortcutNonlinearPolyalg, which
    # target-λ solves the block -> wrong branch. The HomotopyProblem method must continue.
    alg = NLB.initialization_alg(prob, AutoForwardDiff())
    @test alg isa NLB.HomotopyPolyAlgorithm
    @test solve(prob, alg).u[1] ≈ 1.0 atol = 1.0e-6
end
