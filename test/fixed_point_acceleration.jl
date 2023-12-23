using NonlinearSolve, FixedPointAcceleration, LinearAlgebra, Test

# Simple Scalar Problem
@testset "Simple Scalar Problem" begin
    f1(x, p) = cos(x) - x
    prob = NonlinearProblem(f1, 1.1)

    for alg in (:Anderson, :MPE, :RRE, :VEA, :SEA, :Simple, :Aitken, :Newton)
        @test abs(solve(prob, FixedPointAccelerationJL()).resid) ≤ 1e-10
    end
end

# Simple Vector Problem
@testset "Simple Vector Problem" begin
    f2(x, p) = cos.(x) .- x
    prob = NonlinearProblem(f2, [1.1, 1.1])

    for alg in (:Anderson, :MPE, :RRE, :VEA, :SEA, :Simple, :Aitken, :Newton)
        @test maximum(abs.(solve(prob, FixedPointAccelerationJL()).resid)) ≤ 1e-10
    end
end

# Fixed Point for Power Method
# Taken from https://github.com/NicolasL-S/SpeedMapping.jl/blob/95951db8f8a4457093090e18802ad382db1c76da/test/runtests.jl
@testset "Power Method" begin
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
            @test sol.u' * A[:, 3] ≈ 32.916472867168096
        else
            @warn "Power Method failed for FixedPointAccelerationJL(; algorithm = $alg)"
            @test_broken sol.u' * A[:, 3] ≈ 32.916472867168096
        end
    end
end
