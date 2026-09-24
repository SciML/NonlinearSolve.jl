using NonlinearSolve

using SciMLBase
using StaticArrays

# `maxsteps` caps the total number of predictor-corrector attempts. A sweep configured
# to creep (tiny fixed increment, growth disabled) exhausts the cap long before λend
# and must return ReturnCode.MaxIters carrying the last converged (finite) iterate
# instead of looping for span/dλ = 10⁴ steps.
H(u, p, λ) = [u[1] - λ]
prob = HomotopyProblem(H, [0.0]; λspan = (0.0, 1.0))
sol = solve(prob, HomotopySweep(; initial_step_factor = 1.0e-4, expand_factor = 1, maxsteps = 50))
@test sol.retcode == ReturnCode.MaxIters
@test !SciMLBase.successful_retcode(sol)
@test isfinite(sol.u[1])
@test 0 < sol.u[1] < 1
@test sol.resid !== nothing

Hs(u, p, λ) = SA[u[1] - λ]
probs = HomotopyProblem(Hs, SA[0.0]; λspan = (0.0, 1.0))
sol = solve(
    probs,
    SimpleHomotopySweep(; initial_step_factor = 1.0e-4, expand_factor = 1, maxsteps = 50)
)
@test sol.retcode == ReturnCode.MaxIters
@test isfinite(sol.u[1])
@test 0 < sol.u[1] < 1

# a generous maxsteps must not interfere with a normal sweep
sol = solve(prob, HomotopySweep(; maxsteps = 10000))
@test SciMLBase.successful_retcode(sol)
@test sol.u[1] ≈ 1.0 atol = 1.0e-8

@test_throws ArgumentError HomotopySweep(; maxsteps = 0)
@test_throws ArgumentError SimpleHomotopySweep(; maxsteps = 0)

# Effort-band step control: on a smooth easy path (corrector converges in ≤ 2
# iterations, the full-growth band) the controller must still expand the step, taking
# strictly fewer accepted steps than growth disabled — i.e. layering the AUTO-style
# bands under the streak/quality gates does not regress the easy case.
λ_targets = Float64[]
Hg(u, p, λ) = (push!(λ_targets, λ); [u[1] - λ])
probg = HomotopyProblem(Hg, [0.0]; λspan = (0.0, 1.0))
attempts(λs) = [λs[i] for i in eachindex(λs) if i == 1 || λs[i] != λs[i - 1]]

empty!(λ_targets)
sol = solve(probg, HomotopySweep())
@test SciMLBase.successful_retcode(sol)
n_grow = length(attempts(λ_targets))

empty!(λ_targets)
sol = solve(probg, HomotopySweep(; expand_factor = 1))
@test SciMLBase.successful_retcode(sol)
n_nogrow = length(attempts(λ_targets))

# anchor + 10 fixed steps of 0.1 (± one ulp-clamped extra step) without growth; the
# banded controller must still beat that
@test 11 <= n_nogrow <= 12
@test n_grow < n_nogrow

# ArcLengthContinuation with the tracking cap and effort bands still rounds the
# S-curve fold (the item3 problem: path is non-monotone in λ, u ranges the cubic's
# outer sheets), landing on the connected upper-sheet root.
target = 2.1038034
Hf(u, p, λ) = [u[1]^3 - 3 * u[1] - (-3 + 6λ)]
probf = HomotopyProblem(Hf, [-target]; λspan = (0.0, 1.0))
for tm in (100, 20)
    fold_solution = solve(probf, ArcLengthContinuation(; tracking_maxiters = tm))
    @test SciMLBase.successful_retcode(fold_solution)
    @test fold_solution.u[1] ≈ target atol = 1.0e-4
end
