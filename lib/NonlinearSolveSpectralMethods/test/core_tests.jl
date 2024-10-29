@testsetup module CoreRootfindTesting

include("../../../common/common_core_testing.jl")

end

@testitem "DFSane" setup=[CoreRootfindTesting] tags=[:core] begin
    using BenchmarkTools: @ballocated
    using StaticArrays: @SVector

    u0s = ([1.0, 1.0], @SVector[1.0, 1.0], 1.0)

    @testset "[OOP] u0: $(typeof(u0))" for u0 in u0s
        sol = solve_oop(quadratic_f, u0; solver = DFSane())
        @test SciMLBase.successful_retcode(sol)
        err = maximum(abs, quadratic_f(sol.u, 2.0))
        @test err < 1e-9

        cache = init(NonlinearProblem{false}(quadratic_f, u0, 2.0), DFSane(), abstol = 1e-9)
        @test (@ballocated solve!($cache)) < 200
    end

    @testset "[IIP] u0: $(typeof(u0))" for u0 in ([1.0, 1.0],)
        sol = solve_iip(quadratic_f!, u0; solver = DFSane())
        @test SciMLBase.successful_retcode(sol)
        err = maximum(abs, quadratic_f(sol.u, 2.0))
        @test err < 1e-9

        cache = init(NonlinearProblem{true}(quadratic_f!, u0, 2.0), DFSane(), abstol = 1e-9)
        @test (@ballocated solve!($cache)) ≤ 64
    end
end

@testitem "DFSane Iterator Interface" setup=[CoreRootfindTesting] tags=[:core] begin
    p = range(0.01, 2, length = 200)
    @test nlprob_iterator_interface(quadratic_f, p, false, DFSane()) ≈ sqrt.(p)
    @test nlprob_iterator_interface(quadratic_f!, p, true, DFSane()) ≈ sqrt.(p)
end

@testitem "DFSane NewtonRaphson Fails" setup=[CoreRootfindTesting] tags=[:core] begin
    u0 = [-10.0, -1.0, 1.0, 2.0, 3.0, 4.0, 10.0]
    p = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    sol = solve_oop(newton_fails, u0, p; solver = DFSane())
    @test SciMLBase.successful_retcode(sol)
    @test all(abs.(newton_fails(sol.u, p)) .< 1e-9)
end

@testitem "DFSane: Kwargs" setup=[CoreRootfindTesting] tags=[:core] begin
    σ_min = [1e-10, 1e-5, 1e-4]
    σ_max = [1e10, 1e5, 1e4]
    σ_1 = [1.0, 0.5, 2.0]
    M = [10, 1, 100]
    γ = [1e-4, 1e-3, 1e-5]
    τ_min = [0.1, 0.2, 0.3]
    τ_max = [0.5, 0.8, 0.9]
    nexp = [2, 1, 2]
    η_strategy = [
        (f_1, k, x, F) -> f_1 / k^2, (f_1, k, x, F) -> f_1 / k^3,
        (f_1, k, x, F) -> f_1 / k^4
    ]

    list_of_options = zip(σ_min, σ_max, σ_1, M, γ, τ_min, τ_max, nexp, η_strategy)
    for options in list_of_options
        local probN, sol, alg
        alg = DFSane(;
            sigma_min = options[1], sigma_max = options[2], sigma_1 = options[3],
            M = options[4], gamma = options[5], tau_min = options[6],
            tau_max = options[7], n_exp = options[8], eta_strategy = options[9]
        )

        probN = NonlinearProblem{false}(quadratic_f, [1.0, 1.0], 2.0)
        sol = solve(probN, alg, abstol = 1e-11)
        @test all(abs.(quadratic_f(sol.u, 2.0)) .< 1e-6)
    end
end

@testitem "DFSane Termination Conditions" setup=[CoreRootfindTesting] tags=[:core] begin
    using StaticArrays: @SVector

    @testset "TC: $(nameof(typeof(termination_condition)))" for termination_condition in TERMINATION_CONDITIONS
        @testset "u0: $(typeof(u0))" for u0 in ([1.0, 1.0], 1.0, @SVector([1.0, 1.0]))
            probN = NonlinearProblem(quadratic_f, u0, 2.0)
            sol = solve(probN, DFSane(); termination_condition)
            @test all(abs.(quadratic_f(sol.u, 2.0)) .< 1e-10)
        end
    end
end
