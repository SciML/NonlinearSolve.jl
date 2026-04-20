@testitem "Simple Scalar Problem" tags = [:wrappers] begin
    import SpeedMapping, SIAMFANLEquations, NLsolve, FixedPointAcceleration

    f1(x, p) = cos(x) - x
    prob = NonlinearProblem(f1, 1.1)

    for alg in (:Anderson, :MPE, :RRE, :VEA, :SEA, :Simple, :Aitken, :Newton)
        @test abs(solve(prob, FixedPointAccelerationJL(; algorithm = alg)).resid) ≤ 1.0e-10
    end

    @test abs(solve(prob, SpeedMappingJL()).resid) ≤ 1.0e-10
    @test abs(solve(prob, SpeedMappingJL(; orders = [3, 2])).resid) ≤ 1.0e-10
    @test abs(solve(prob, SpeedMappingJL(; stabilize = true)).resid) ≤ 1.0e-10

    @test abs(solve(prob, NLsolveJL(; method = :anderson)).resid) ≤ 1.0e-10
    @test abs(solve(prob, SIAMFANLEquationsJL(; method = :anderson)).resid) ≤ 1.0e-10
end

# Simple Vector Problem
@testitem "Simple Vector Problem" tags = [:wrappers] begin
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
end

# Fixed Point for Power Method
# Taken from https://github.com/NicolasL-S/SpeedMapping.jl/blob/95951db8f8a4457093090e18802ad382db1c76da/test/runtests.jl
@testitem "Power Method" tags = [:wrappers] begin
    using LinearAlgebra
    import SpeedMapping, SIAMFANLEquations, NLsolve, FixedPointAcceleration

    C = [1 2 3; 4 5 6; 7 8 9]
    A = C + C'
    B = Hermitian(ones(10) * ones(10)' .* im + Diagonal(1:10))

    function power_method!(du, u, A)
        mul!(du, A, u)
        du ./= norm(du, Inf)
        du .-= u  # Convert to a root finding problem
        return nothing
    end

    prob = NonlinearProblem(power_method!, ones(3), A)

    for alg in (:Anderson, :MPE, :RRE, :VEA, :SEA, :Simple, :Aitken, :Newton)
        sol = solve(prob, FixedPointAccelerationJL(; algorithm = alg))
        if SciMLBase.successful_retcode(sol)
            @test sol.u' * A[:, 3] ≈ 32.916472867168096
        else
            @warn "Power Method failed for FixedPointAccelerationJL(; algorithm = $alg)"
            @test_broken sol.u' * A[:, 3] ≈ 32.916472867168096
        end
    end

    for kwargs in ((;), (; orders = [3, 2]), (; stabilize = true))
        alg = SpeedMappingJL(; kwargs...)
        sol = solve(prob, alg)
        @test sol.u' * A[:, 3] ≈ 32.916472867168096
    end

    sol = solve(prob, NLsolveJL(; method = :anderson))
    @test_broken sol.u' * A[:, 3] ≈ 32.916472867168096

    sol = solve(prob, SIAMFANLEquationsJL(; method = :anderson))
    @test sol.u' * A[:, 3] ≈ 32.916472867168096

    # Non vector inputs
    function power_method_nonvec!(du, u, A)
        mul!(vec(du), A, vec(u))
        du ./= norm(du, Inf)
        du .-= u  # Convert to a root finding problem
        return nothing
    end

    prob = NonlinearProblem(power_method_nonvec!, ones(1, 3, 1), A)

    for alg in (:Anderson, :MPE, :RRE, :VEA, :SEA, :Simple, :Aitken, :Newton)
        sol = solve(prob, FixedPointAccelerationJL(; algorithm = alg))
        if SciMLBase.successful_retcode(sol)
            @test vec(sol.u)' * A[:, 3] ≈ 32.916472867168096
        else
            @warn "Power Method failed for FixedPointAccelerationJL(; algorithm = $alg)"
            @test_broken vec(sol.u)' * A[:, 3] ≈ 32.916472867168096
        end
    end

    for kwargs in ((;), (; orders = [3, 2]), (; stabilize = true))
        alg = SpeedMappingJL(; kwargs...)
        sol = solve(prob, alg)
        @test vec(sol.u)' * A[:, 3] ≈ 32.916472867168096
    end

    sol = solve(prob, NLsolveJL(; method = :anderson))
    @test_broken vec(sol.u)' * A[:, 3] ≈ 32.916472867168096
end

# Issue #862: NLsolveJL and SIAMFANLEquationsJL Anderson must NOT allocate a dense
# N×N Jacobian (Anderson acceleration is Jacobian-free).
@testitem "Anderson does not allocate dense Jacobian (#862)" tags = [:wrappers] begin
    import NLsolve, SIAMFANLEquations

    # N large enough that a dense N×N Float64 matrix would dominate the
    # measured allocation, but small enough to solve quickly.
    N = 3_200
    # Simple fixed point: f(x) = 0 at x_i = 0.739... (cos(x) = x).
    F!(F, x, p) = (F .= cos.(x) .- x)
    x0 = fill(0.5, N)
    prob = NonlinearProblem(NonlinearFunction(F!), x0)

    # Cap iterations so per-iteration allocations don't dwarf the one-shot
    # Jacobian allocation we're guarding against.
    maxit = 10

    # Warm up compilation
    solve(prob, NLsolveJL(; method = :anderson, m = 5); maxiters = maxit)
    solve(prob, SIAMFANLEquationsJL(; method = :anderson, m = 5); maxiters = maxit)

    # An N×N Float64 matrix is 8·N² bytes ≈ 82 MB at N=3200.
    dense_jac_bytes = 8 * N * N

    allocs_nlsolve = @allocated solve(
        prob, NLsolveJL(; method = :anderson, m = 5); maxiters = maxit
    )
    @test allocs_nlsolve < dense_jac_bytes ÷ 4

    allocs_siam = @allocated solve(
        prob, SIAMFANLEquationsJL(; method = :anderson, m = 5); maxiters = maxit
    )
    @test allocs_siam < dense_jac_bytes ÷ 4
end
