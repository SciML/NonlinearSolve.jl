using NonlinearSolve, SciMLBase, Test

# The λ = 0 anchor starts at an exact root, so every polyalgorithm subcache is initialized
# with a zero residual. Retention then skips the cheap rungs while tracking, and the sweep
# stalls at the fold (λ, u) = (5/6, -1): the failed corrector must report its own
# residual and return code, not those of a subcache last run at the anchor.
@testset "sweep stalled at a fold reports the failing corrector, not a stale subcache" begin
    H(u, p, λ) = [u[1]^3 - 3u[1] - (-3 + 6λ)]
    u0 = [-2.1038034027355366]
    prob = HomotopyProblem(H, u0, nothing; λspan = (0.0, 1.0))
    sol = solve(prob, HomotopySweep(; store_original = Val(true)))
    @test !SciMLBase.successful_retcode(sol)
    @test sol.retcode != ReturnCode.Default
    @test sol.u[1] ≈ -1 atol = 1.0e-2
    @test sol.original.retcode != ReturnCode.Default
    @test sol.original.u != u0
end
