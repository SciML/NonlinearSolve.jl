using NonlinearSolve

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
