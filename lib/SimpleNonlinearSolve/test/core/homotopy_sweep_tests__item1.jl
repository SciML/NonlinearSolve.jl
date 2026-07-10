using SimpleNonlinearSolve, StaticArrays, SciMLBase

# ---- construction + validation ----
alg = SimpleHomotopySweep()
@test alg isa SimpleHomotopySweep
@test alg.inner isa SimpleNewtonRaphson
@test alg.adaptive
@test alg.predictor === :secant

@test_throws ArgumentError SimpleHomotopySweep(; nsteps = 0)
@test_throws ArgumentError SimpleHomotopySweep(; adaptive = false)
@test_throws ArgumentError SimpleHomotopySweep(; initial_step_factor = 0.0)
@test_throws ArgumentError SimpleHomotopySweep(; min_dλ = 0.0)
@test_throws ArgumentError SimpleHomotopySweep(; max_step_factor = 1.5)
@test_throws ArgumentError SimpleHomotopySweep(; expand_factor = 0.5)
@test_throws ArgumentError SimpleHomotopySweep(; expand_threshold = 0)
@test_throws ArgumentError SimpleHomotopySweep(; expand_quality = 0.0)
@test_throws ArgumentError SimpleHomotopySweep(; predictor = :tangent)
@test_throws ArgumentError SimpleHomotopySweep(; tracking_abstol = 0.0)
@test_throws ArgumentError SimpleHomotopySweep(; tracking_abstol = -1.0e-3)

# ---- correctness across container types ----
# (1-λ)(u - p) + λ(u² - p): root path from u = p at λ = 0 to u = √p at λ = 1
H(u, p, λ) = @. (1 - λ) * (u - p) + λ * (u^2 - p)

sol_v = solve(HomotopyProblem(H, [4.0], 4.0), SimpleHomotopySweep())
@test SciMLBase.successful_retcode(sol_v)
@test sol_v.u[1] ≈ 2.0 atol = 1.0e-6

sol_s = solve(HomotopyProblem(H, SA[4.0], 4.0), SimpleHomotopySweep())
@test SciMLBase.successful_retcode(sol_s)
@test sol_s.u isa SVector{1, Float64}
@test sol_s.u[1] ≈ 2.0 atol = 1.0e-6

sol_32 = solve(
    HomotopyProblem(H, SA[4.0f0], 4.0f0; λspan = (0.0f0, 1.0f0)), SimpleHomotopySweep()
)
@test SciMLBase.successful_retcode(sol_32)
@test eltype(sol_32.u) == Float32
@test sol_32.u[1] ≈ 2.0f0 atol = 1.0f-4

H!(du, u, p, λ) = (du .= (1 .- λ) .* (u .- p) .+ λ .* (u .^ 2 .- p); nothing)
sol_i = solve(HomotopyProblem(NonlinearFunction{true}(H!), [4.0], 4.0), SimpleHomotopySweep())
@test SciMLBase.successful_retcode(sol_i)
@test sol_i.u[1] ≈ 2.0 atol = 1.0e-6

# ---- decreasing λspan ----
sol_dec = solve(
    HomotopyProblem(H, SA[2.0], 4.0; λspan = (1.0, 0.0)), SimpleHomotopySweep()
)
@test SciMLBase.successful_retcode(sol_dec)
@test sol_dec.u[1] ≈ 4.0 atol = 1.0e-6      # λ = 0 system is linear: u = p

# ---- failure honesty: fold with no real root past λ = 1/3 must fail, not "succeed" ----
Hfold(u, p, λ) = @. (1 - λ) * u + λ * (u^2 + 1)
sol_f = solve(
    HomotopyProblem(Hfold, SA[0.0]; λspan = (0.0, 1.0)),
    SimpleHomotopySweep(; min_dλ = 1.0e-2)
)
@test !SciMLBase.successful_retcode(sol_f)
@test all(isfinite, sol_f.u)                # last converged iterate, not diverged junk

# ---- adaptive expansion takes fewer accepted steps than fixed increments ----
λ_targets = Float64[]
Hcount(u, p, λ) = (push!(λ_targets, λ); [u[1] - λ])
prob_count = HomotopyProblem(Hcount, [0.0]; λspan = (0.0, 1.0))
attempts(λs) = [λs[i] for i in eachindex(λs) if i == 1 || λs[i] != λs[i - 1]]

empty!(λ_targets)
solve(prob_count, SimpleHomotopySweep())
n_grow = length(attempts(λ_targets))
empty!(λ_targets)
solve(prob_count, SimpleHomotopySweep(; expand_factor = 1))
n_nogrow = length(attempts(λ_targets))
@test n_grow < n_nogrow

# ---- fixed-step mode ----
sol_fix = solve(
    HomotopyProblem(H, SA[4.0], 4.0), SimpleHomotopySweep(; adaptive = false, nsteps = 10)
)
@test SciMLBase.successful_retcode(sol_fix)
@test sol_fix.u[1] ≈ 2.0 atol = 1.0e-6
