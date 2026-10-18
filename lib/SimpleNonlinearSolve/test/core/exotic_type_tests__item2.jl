using SimpleNonlinearSolve, SciMLBase, StaticArrays, InteractiveUtils

prob = convert(
    SciMLBase.ImmutableNonlinearProblem,
    NonlinearProblem{false}((u, p) -> u .* u .- p, SVector(1.0f0, 1.0f0), 2.0f0)
)
solve_u(prob, alg) = solve(prob, alg; abstol = 1.0f-6, reltol = 1.0f-6).u

@testset "nlsolve_update_rule = $update" for update in (Val(false), Val(true))
    alg = SimpleTrustRegion(; nlsolve_update_rule = update)
    @test !any(i -> getfield(alg, i) isa AbstractFloat, 1:fieldcount(typeof(alg)))
    ir = sprint(io -> code_llvm(io, solve_u, Tuple{typeof(prob), typeof(alg)}))
    @test !occursin(r"\bdouble\b", ir)
    u = solve_u(prob, alg)
    @test u isa SVector{2, Float32}
    @test u ≈ fill(sqrt(2.0f0), 2)
end

@test solve_u(prob, SimpleTrustRegion(; max_trust_radius = 10.0, expand_factor = 3.0)) ≈
    fill(sqrt(2.0f0), 2)
