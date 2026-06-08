using NonlinearSolve

import SpeedMapping, SIAMFANLEquations, NLsolve, FixedPointAcceleration

f2(x, p) = cos.(x) .- x
prob = NonlinearProblem(f2, [1.1, 1.1])

for alg in (:Anderson, :MPE, :RRE, :VEA, :SEA, :Simple, :Aitken, :Newton)
    @test maximum(abs.(solve(prob, FixedPointAccelerationJL()).resid)) ≤ 1.0e-10
end

@test maximum(abs.(solve(prob, SpeedMappingJL()).resid)) ≤ 1.0e-10
@test maximum(abs.(solve(prob, SpeedMappingJL(; orders = [3, 2])).resid)) ≤ 1.0e-10
@test maximum(abs.(solve(prob, SpeedMappingJL(; stabilize = true)).resid)) ≤ 1.0e-10
@test maximum(abs.(solve(prob, SIAMFANLEquationsJL(; method = :anderson)).resid)) ≤
    1.0e-10

@test_broken maximum(abs.(solve(prob, NLsolveJL(; method = :anderson)).resid)) ≤ 1.0e-10
