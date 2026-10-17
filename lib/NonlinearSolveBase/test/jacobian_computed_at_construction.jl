using NonlinearSolveBase, SciMLBase, ADTypes, ForwardDiff, LinearSolve, LinearAlgebra, Test
using NonlinearSolveBase: construct_jacobian_cache, jacobian_computed_at_construction,
    reused_jacobian

const N = 4
const A = [1 / (1 + abs(i - j)) for i in 1:N, j in 1:N] + 2N * I
f!(du, u, p) = (mul!(du, A, u); du .+= sin.(u) .- p; nothing)
f(u, p) = A * u .+ sin.(u) .- p
jac!(J, u, p) = (J .= A .+ Diagonal(cos.(u)); nothing)
true_jac(u) = A + Diagonal(cos.(u))

u0 = collect(range(0.1, 0.7; length = N))
stats() = SciMLBase.NLStats(0, 0, 0, 0, 0)

function jac_cache(nf; linsolve = nothing, kwargs...)
    prob = NonlinearProblem(nf, u0, 1.0)
    fu = SciMLBase.isinplace(prob) ? (du = similar(u0); f!(du, u0, 1.0); du) : f(u0, 1.0)
    return construct_jacobian_cache(
        prob, nothing, prob.f, fu; stats = stats(), autodiff = AutoForwardDiff(), linsolve,
        kwargs...
    )
end

@testset "evaluated at construction" begin
    for nf in (NonlinearFunction{true}(f!), NonlinearFunction{false}(f))
        cache = jac_cache(nf)
        @test jacobian_computed_at_construction(cache)
        @test reused_jacobian(cache, u0) ≈ true_jac(u0)
    end
end

@testset "only allocated at construction" begin
    @test !jacobian_computed_at_construction(
        jac_cache(NonlinearFunction{true}(f!; jac = jac!))
    )
    @test !jacobian_computed_at_construction(
        jac_cache(NonlinearFunction{true}(f!; jac_prototype = zeros(N, N)))
    )
    @test !jacobian_computed_at_construction(
        jac_cache(
            NonlinearFunction{true}(f!); linsolve = KrylovJL_GMRES(),
            jvp_autodiff = AutoForwardDiff(), vjp_autodiff = AutoForwardDiff()
        )
    )

    scalar_prob = NonlinearProblem((u, p) -> u^2 - p, 1.5, 2.0)
    scalar_cache = construct_jacobian_cache(
        scalar_prob, nothing, scalar_prob.f, 0.25; stats = stats(),
        autodiff = AutoForwardDiff()
    )
    @test !jacobian_computed_at_construction(scalar_cache)

    @test !jacobian_computed_at_construction(nothing)
end
