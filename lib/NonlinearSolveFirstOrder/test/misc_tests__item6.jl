using NonlinearSolveFirstOrder

# The `corrector` post-update hook runs at the seam between the Newton update
# (u ← u + αδu) and the residual re-evaluation. It is called as
# `corrector(u_proposed, u_prev, p)`, mutates `u` in place, and the corrected
# `u` is what the subsequent residual and termination check observe.
#
# Slaved-variable system (the shape SPICE junction limiting / PCNR produces):
# u[1] is free, u[2] is slaved to u[1]^3 by a linear tracking row. A corrector
# that overwrites u[2] := u[1]^3 keeps that row satisfied right after the step.
function f!(F, u, p)
    F[1] = u[1] - p[1]
    F[2] = u[2] - u[1]^3
    return nothing
end

u0 = [0.0, 0.0]
p = [1.0]
prob = NonlinearProblem(NonlinearFunction(f!), u0, p)

# --- corrector invoked with the right arguments, in the right order ---
calls = Vector{Tuple{Vector{Float64}, Vector{Float64}}}()
function corrector!(u, u_prev, p)
    push!(calls, (copy(u), copy(u_prev)))  # record BEFORE mutating
    u[2] = u[1]^3
    return nothing
end

sol = solve(prob, NewtonRaphson(; corrector = corrector!); abstol = 1e-12)

@test sol.retcode == ReturnCode.Success
@test sol.u[1] ≈ 1.0 atol = 1e-10
@test sol.u[2] ≈ 1.0 atol = 1e-10
# the corrector enforced the slaving exactly on the returned iterate
@test sol.u[2] == sol.u[1]^3
# called exactly once per Newton step
@test length(calls) == sol.stats.nsteps
@test sol.stats.nsteps ≥ 1
# ordering: the `u` handed to the corrector is the POST-update iterate — on the
# first step it differs from `u_prev`, which is the initial guess `u0`
u_first, uprev_first = calls[1]
@test uprev_first == u0
@test u_first != u0                       # axpy! ran before the corrector fired
# every subsequent call sees the slaving the previous call applied
for (_, up) in calls[2:end]
    @test up[2] == up[1]^3
end

# --- `nothing` (the default) is a genuine no-op ---
sol_default = solve(prob, NewtonRaphson(); abstol = 1e-12)
sol_nothing = solve(prob, NewtonRaphson(; corrector = nothing); abstol = 1e-12)
@test sol_nothing.u == sol_default.u
@test sol_nothing.stats.nsteps == sol_default.stats.nsteps

# an inert (identity) corrector must not change the result vs no corrector
inert_corrector!(u, u_prev, p) = nothing
sol_inert = solve(prob, NewtonRaphson(; corrector = inert_corrector!); abstol = 1e-12)
@test sol_inert.u ≈ sol_default.u
@test sol_inert.stats.nsteps == sol_default.stats.nsteps

# the slaving corrector converges the augmented system in fewer steps than plain
# Newton (it removes the slaved variable's lag) — evidence the hook does work
@test sol.stats.nsteps < sol_default.stats.nsteps
