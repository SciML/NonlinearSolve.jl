using SimpleNonlinearSolve, StaticArrays, BenchmarkTools, DiffEqBase, LinearAlgebra, Test,
    NNlib, AbstractDifferentiation, LinearSolve

# Supported Solvers: BatchedBroyden, SimpleBatchedDFSane
function f!(du::AbstractArray{<:Number, N},
    u::AbstractArray{<:Number, N},
    p::AbstractVector) where {N}
    u_ = reshape(u, :, size(u, N))
    du .= reshape(sum(abs2, u_; dims = 1) .- reshape(p, 1, :),
        ntuple(_ -> 1, N - 1)...,
        size(u, N))
    return du
end

function f!(du::AbstractMatrix, u::AbstractMatrix, p::AbstractVector)
    du .= sum(abs2, u; dims = 1) .- reshape(p, 1, :)
    return du
end

function f!(du::AbstractVector, u::AbstractVector, p::AbstractVector)
    du .= sum(abs2, u) .- p
    return du
end

@testset "Solver: $(nameof(typeof(solver)))" for solver in (Broyden(batched = true),
    SimpleDFSane(batched = true))
    @testset "T: $T" for T in (Float32, Float64)
        p = rand(T, 5)
        @testset "size(u0): $sz" for sz in ((2, 5), (1, 5), (2, 3, 5))
            u0 = ones(T, sz)
            prob = NonlinearProblem{true}(f!, u0, p)

            sol = solve(prob, solver)

            @test SciMLBase.successful_retcode(sol.retcode)

            @test sol.resid≈zero(sol.resid) atol=5e-3
        end

        p = rand(T, 1)
        @testset "size(u0): $sz" for sz in ((3,), (5,), (10,))
            u0 = ones(T, sz)
            prob = NonlinearProblem{true}(f!, u0, p)

            sol = solve(prob, solver)

            @test SciMLBase.successful_retcode(sol.retcode)

            @test sol.resid≈zero(sol.resid) atol=5e-3
        end
    end
end
