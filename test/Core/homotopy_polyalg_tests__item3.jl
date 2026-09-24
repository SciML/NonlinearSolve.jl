using NonlinearSolve

using SciMLBase

# A `HomotopyProblem` handed a *standard* (non-continuation) nonlinear algorithm solves its
# target-λ system (`λ = λspan[2]`) directly, by fixing λ — the requested algorithm is
# honored, exactly as if it were the equivalent `NonlinearProblem`. Continuation stays
# opt-in via `nothing`, `HomotopySweep`, `ArcLengthContinuation`, or `HomotopyPolyAlgorithm`.

# --- target-λ solve equals solving the equivalent NonlinearProblem with the same alg ---
# path (1-λ)(u-2) + λ(u²-4); at λ = 1 the target system is u² - 4 (roots ±2).
Hquad(u, p, λ) = [(1 - λ) * (u[1] - 2) + λ * (u[1]^2 - 4)]
hp = HomotopyProblem(Hquad, [1.5]; λspan = (0.0, 1.0))
np = NonlinearProblem((u, p) -> [u[1]^2 - 4], [1.5])
for alg in (NewtonRaphson(), TrustRegion(), FastShortcutNonlinearPolyalg())
    hs = solve(hp, alg; abstol = 1.0e-10)
    ns = solve(np, alg; abstol = 1.0e-10)
    @test SciMLBase.successful_retcode(hs)
    @test hs.u ≈ ns.u atol = 1.0e-8         # same result as the plain NonlinearProblem
    @test hs.u[1] ≈ 2.0 atol = 1.0e-8
end

# --- a custom λspan targets λspan[2], not necessarily 1 ---
# at λ = 0.5 the target residual is 0.5(u-2) + 0.5(u²-4) = 0 -> u² + u - 6 = 0 -> u = 2
hp_half = HomotopyProblem(Hquad, [1.5]; λspan = (0.0, 0.5))
np_half = NonlinearProblem((u, p) -> [0.5 * (u[1] - 2) + 0.5 * (u[1]^2 - 4)], [1.5])
@test solve(hp_half, NewtonRaphson(); abstol = 1.0e-10).u ≈
    solve(np_half, NewtonRaphson(); abstol = 1.0e-10).u atol = 1.0e-8

# --- continuation stays opt-in ---
# `atan` saturates, so a plain Newton on the λ = 1 system (atan(u-3) = 0) from u0 = 12 does
# not reach the root — the standard-algorithm path does exactly the plain-NonlinearProblem
# thing, whatever that is, while `nothing` and the homotopy algorithms sweep and rescue it.
Hatan(u, p, λ) = [(1 - λ) * u[1] + λ * atan(u[1] - 3)]
hp_atan = HomotopyProblem(Hatan, [12.0]; λspan = (0.0, 1.0))
np_atan = NonlinearProblem((u, p) -> [atan(u[1] - 3)], [12.0])
@test solve(hp_atan, NewtonRaphson()).u ≈ solve(np_atan, NewtonRaphson()).u atol = 1.0e-8
for alg in (nothing, HomotopySweep(), ArcLengthContinuation(), HomotopyPolyAlgorithm())
    s = alg === nothing ? solve(hp_atan) : solve(hp_atan, alg)
    @test SciMLBase.successful_retcode(s)
    @test s.u[1] ≈ 3.0 atol = 1.0e-6
end
