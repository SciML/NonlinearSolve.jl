using NonlinearSolve

using SciMLBase

# Linear-in-u homotopy: every step's corrector succeeds, so the distinct λ values the
# residual is called with are exactly the accepted continuation steps, in order.
λ_targets = Float64[]
H(u, p, λ) = (push!(λ_targets, λ); [u[1] - λ])
prob = HomotopyProblem(H, [0.0]; λspan = (0.0, 1.0))

# f is evaluated many times per step; collapse consecutive repeats of the same λ
function attempts(λs)
    return [λs[i] for i in eachindex(λs) if i == 1 || λs[i] != λs[i - 1]]
end

empty!(λ_targets)
sol = solve(prob, HomotopySweep())
@test SciMLBase.successful_retcode(sol)
@test sol.u[1] ≈ 1.0 atol = 1.0e-8
n_grow = length(attempts(λ_targets))

empty!(λ_targets)
sol = solve(prob, HomotopySweep(; expand_factor = 1))
@test SciMLBase.successful_retcode(sol)
@test sol.u[1] ≈ 1.0 atol = 1.0e-8
n_nogrow = length(attempts(λ_targets))

# expand_factor = 1 reproduces the fixed initial increment: 10 steps of 0.1 (the
# accumulated 0.1's can undershoot λend by one ulp, costing one extra clamped step)
@test 10 <= n_nogrow <= 11
# expansion (defaults: ×2 every 2 successes) must take strictly fewer steps
@test n_grow < n_nogrow

# the expanded increment must never exceed max_step_factor of the span
empty!(λ_targets)
sol = solve(prob, HomotopySweep(; expand_threshold = 1, max_step_factor = 0.2))
@test SciMLBase.successful_retcode(sol)
steps = diff(vcat(0.0, attempts(λ_targets)))
@test maximum(steps) <= 0.2 + 1.0e-12

# nsteps sets the initial increment but the cap still binds it
empty!(λ_targets)
sol = solve(prob, HomotopySweep(; nsteps = 2, max_step_factor = 0.25))
@test SciMLBase.successful_retcode(sol)
steps = diff(vcat(0.0, attempts(λ_targets)))
@test maximum(steps) <= 0.25 + 1.0e-12
