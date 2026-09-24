using NonlinearSolve

using SciMLBase

# Regression: `reinit!` with only a new `u0` (parameters unchanged) must preserve the
# cache's parameter *type*, not reset it to `missing`. Rebuilding the internal
# `LinearSolveParameters` with a `Missing` p-field mismatched the concretely-typed `p`
# (e.g. `NullParameters`) the LinearSolve-backed cache was built with, throwing a
# `setfield!` type error for every solver that uses that linear cache (NewtonRaphson,
# TrustRegion, and the default polyalgorithm) — so a continuation-style loop that inits
# once and `reinit!`s each step could not be built.
n = 5
f(u, p) = u .^ 2 .- p
prob = NonlinearProblem(f, ones(n), fill(4.0, n))

for alg in (NewtonRaphson(), TrustRegion(), nothing)
    cache = init(prob, alg)
    s1 = solve!(cache)
    @test SciMLBase.successful_retcode(s1)
    @test s1.u ≈ fill(2.0, n) atol = 1.0e-6

    # only a new warm start, parameters left unchanged — must not throw, root unchanged
    SciMLBase.reinit!(cache, fill(0.5, n))
    s2 = solve!(cache)
    @test SciMLBase.successful_retcode(s2)
    @test s2.u ≈ fill(2.0, n) atol = 1.0e-6

    # an explicit new `p` must still be applied (root moves to sqrt(9) = 3)
    SciMLBase.reinit!(cache, ones(n); p = fill(9.0, n))
    s3 = solve!(cache)
    @test SciMLBase.successful_retcode(s3)
    @test s3.u ≈ fill(3.0, n) atol = 1.0e-6
end
