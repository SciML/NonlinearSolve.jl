using NonlinearSolve, SpeedMapping, LinearAlgebra, Test

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

    sol = solve(prob, SpeedMappingJL())
    @test sol.u' * A[:, 3] ≈ 32.916472867168096

    sol = solve(prob, SpeedMappingJL(; orders = [3, 2]))
    @test sol.u' * A[:, 3] ≈ 32.916472867168096

    sol = solve(prob, SpeedMappingJL(; stabilize = true))
    @test sol.u' * A[:, 3] ≈ 32.91647286145264
end
