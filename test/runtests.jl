using NonlinearSolve
using StaticArrays
using BenchmarkTools
using Test

function benchmark_immutable()
    probN = NonlinearProblem((u,p) -> u .* u .- 2, @SVector[1.0, 1.0])
    solver = init(probN, NewtonRaphson(), immutable = true, tol = 1e-9)
    sol = @btime solve!($solver)
    @test all(sol.u .* sol.u .- 2 .< 1e-9)
end

function benchmark_mutable()
    probN = NonlinearProblem((u,p) -> u .* u .- 2, @SVector[1.0, 1.0])
    solver = init(probN, NewtonRaphson(), immutable = false, tol = 1e-9)
    sol = @btime (reinit!($solver, $probN); solve!($solver))
    @test all(sol.u .* sol.u .- 2 .< 1e-9)
end

function benchmark_scalar()
    probN = NonlinearProblem((u,p) -> u .* u .- 2, 1.0)
    sol = @btime (solve($probN, ScalarNewton()))
    @test sol * sol - 2 < 1e-9
end

benchmark_immutable()
benchmark_mutable()
benchmark_scalar()
