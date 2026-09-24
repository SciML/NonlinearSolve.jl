using SCCNonlinearSolve
using NonlinearSolve
using SciMLBase
using Test

# A `HomotopyProblem` block of an `SCCNonlinearProblem` is solved by continuation (using the
# SCC's nonlinear algorithm as the inner corrector), not by that algorithm applied directly
# to the block's target-λ system.

# branch selection: actual m² - 1 has roots ±1; from the guess m = -3 a Newton solve of the
# target (λ = 1) system lands on the UNPHYSICAL -1, while continuation from the simplified
# 2(m-1) (whose anchor is +1) reaches the physical +1. So a passing test *requires* the block
# to have been continued.
Hbranch(u, p, λ) = [(1 - λ) * 2 * (u[1] - 1) + λ * (u[1]^2 - 1)]
homblock = HomotopyProblem(Hbranch, [-3.0])
nlblock = NonlinearProblem((u, p) -> [u[1]^2 - 4], [1.5])   # plain block, root 2

# control: the algorithm applied directly to the block solves the target system -> -1
@test solve(homblock, NewtonRaphson()).u[1] ≈ -1.0 atol = 1.0e-6

@testset "homotopy SCC block is continued, not target-λ solved" begin
    sccprob = SciMLBase.SCCNonlinearProblem(
        (homblock, nlblock),
        SciMLBase.Void{Any}.([Returns(nothing), Returns(nothing)])
    )
    sol = solve(sccprob, SCCNonlinearSolve.SCCAlg(nlalg = NewtonRaphson(), linalg = nothing))
    @test sol[1] ≈ 1.0 atol = 1.0e-6      # homotopy block continued to the physical root
    @test sol[2] ≈ 2.0 atol = 1.0e-6      # plain block
end

@testset "the inner corrector's autodiff is honored" begin
    # a ForwardDiff-hostile residual can only be finite-differenced; passing a finite-diff
    # Newton as the SCC algorithm must thread that AD into the continuation's inner corrector.
    badf(x::Float64) = x^2 - 1
    badf(x) = throw(ArgumentError("residual differentiated by ForwardDiff (dual seen)"))
    Hbad(u, p, λ) = [(1 - λ) * 2 * (u[1] - 1) + λ * badf(u[1])]
    sccprob = SciMLBase.SCCNonlinearProblem(
        (HomotopyProblem(Hbad, [-3.0]),), SciMLBase.Void{Any}.([Returns(nothing)])
    )
    sol_fd = solve(
        sccprob,
        SCCNonlinearSolve.SCCAlg(nlalg = NewtonRaphson(autodiff = AutoFiniteDiff()), linalg = nothing)
    )
    @test sol_fd[1] ≈ 1.0 atol = 1.0e-6

    threw = false
    try
        solve(
            sccprob,
            SCCNonlinearSolve.SCCAlg(nlalg = NewtonRaphson(autodiff = AutoForwardDiff()), linalg = nothing)
        )
    catch
        threw = true
    end
    @test threw
end
