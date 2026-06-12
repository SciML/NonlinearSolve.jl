using NonlinearSolve

using SciMLBase

# The path u*(λ) = 3tanh(20(λ - 0.5)) turns sharply at λ = 0.5, and the residual
# x + 2sin(x) (x = u - path) has spurious merit-function stationary points at
# |x| ≈ 4.19 that trap correctors launched from |x| > 2.09, so oversized steps across
# the turn are rejected. The sweep must bisect through the turn and then regrow the
# increment on the flat shoulder past it.
λ_targets = Float64[]
function H(u, p, λ)
    push!(λ_targets, λ)
    x = u[1] - 3 * tanh(20 * (λ - 0.5))
    return [x + 2 * sin(x)]
end
prob = HomotopyProblem(H, [-3 * tanh(10.0)]; λspan = (0.0, 1.0))

# consecutive dedup: f is evaluated many times per attempted λ; a rejected λ can also
# be revisited and accepted later, so a global unique() would conflate the two
function attempts(λs)
    return [λs[i] for i in eachindex(λs) if i == 1 || λs[i] != λs[i - 1]]
end

empty!(λ_targets)
sol = solve(prob, HomotopySweep(; initial_step_factor = 0.25); maxiters = 100)
@test SciMLBase.successful_retcode(sol)
@test sol.u[1] ≈ 3 * tanh(10.0) atol = 1.0e-6
seq = attempts(λ_targets)
# an attempt is rejected iff the next attempt retreats below it
rejected = [i for i in 1:(length(seq) - 1) if seq[i + 1] < seq[i]]
@test !isempty(rejected)
# accepted increments after the last rejection: the bisected step must regrow
accepted = seq[(last(rejected) + 1):end]
steps = diff(accepted)
@test all(>(0), steps)
@test maximum(steps) >= 2 * minimum(steps)
n_grow = length(seq)

empty!(λ_targets)
sol = solve(
    prob, HomotopySweep(; initial_step_factor = 0.25, expand_factor = 1);
    maxiters = 100
)
@test SciMLBase.successful_retcode(sol)
@test sol.u[1] ≈ 3 * tanh(10.0) atol = 1.0e-6
# without regrowth the sweep stays at the bisected resolution for the rest of the
# span, taking strictly more steps
@test n_grow < length(attempts(λ_targets))
