using NonlinearSolveFirstOrder, NonlinearSolveBase, SciMLBase, LinearAlgebra, Test

f!(du, u, p) = (du .= u .* u .- p)
j!(J, u, p) = (
    fill!(J, 0); for i in eachindex(u)
        J[i, i] = 2 * u[i]
    end; J
)

function warmed_active_step_bytes(alg)
    prob = NonlinearProblem(
        NonlinearFunction(f!; jac = j!), [0.1, 4.0], [2.0, 3.0]
    )
    bytes = Int[]
    for _ in 1:4
        cache = init(prob, alg; abstol = 1.0e-10, reltol = 1.0e-10, maxiters = 100)
        step!(cache)
        @test !cache.force_stop  # still active after the first step
        push!(bytes, @allocated step!(cache))
    end
    return minimum(bytes)
end

@testset "Warmed active in-place TrustRegion (Moré) step is allocation-free" begin
    @test warmed_active_step_bytes(TrustRegion()) == 0
end

@testset "Warmed active in-place LM damping reuses the diagonal buffer" begin
    # Master baseline for this probe: 528 with geodesic, 352 without.
    @test warmed_active_step_bytes(LevenbergMarquardt()) ≤ 528
    @test warmed_active_step_bytes(LevenbergMarquardt(; disable_geodesic = Val(true))) ≤ 352
end
