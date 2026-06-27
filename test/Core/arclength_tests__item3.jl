using NonlinearSolve

using SciMLBase

# S-shaped fold. The curve u^3 - 3u = μ(λ), μ(λ) = -3 + 6λ, has folds (∂H/∂u = 0) at
# u = ±1, i.e. μ = ±2, i.e. λ = 5/6 and λ = 1/6. Continuing the connected branch from the
# λ = 0 root on the lower sheet (u ≈ -2.1038) to the λ = 1 root on the upper sheet
# (u ≈ +2.1038) requires λ to first rise to 5/6, reverse down to 1/6, then rise to 1 —
# the path is non-monotone in λ. Natural-parameter marching cannot reverse λ; arclength
# continuation tracks the augmented curve through both turning points.
target = 2.1038034
λseen = Float64[]
function H(u, p, λ)
    # record only the plain-Float64 residual evaluations; the corrector's Jacobian passes
    # ForwardDiff Duals through λ, which we must not push into a Float64 buffer
    λ isa Float64 && push!(λseen, λ)
    return [u[1]^3 - 3 * u[1] - (-3 + 6λ)]
end
prob = HomotopyProblem(H, [-target]; λspan = (0.0, 1.0))

empty!(λseen)
sol = solve(prob, ArcLengthContinuation())
@test SciMLBase.successful_retcode(sol)
# lands on the connected upper-sheet root at λ = 1, with a genuine target-system residual
@test sol.u[1] ≈ target atol = 1.0e-4
@test abs(sol.u[1]^3 - 3 * sol.u[1] - 3) < 1.0e-6

# proof the fold was rounded: λ rises past the first turning point (> 0.8) and then
# decreases well below it (toward the second turning point at 1/6) before reaching 1.
i_past_fold = findfirst(>(0.8), λseen)
@test i_past_fold !== nothing
@test minimum(λseen[i_past_fold:end]) < 0.3
