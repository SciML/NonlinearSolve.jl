using NonlinearSolveFirstOrder, LinearSolve
using LinearAlgebra, SparseArrays, SparseMatrixColorings, Test

# Moré normal-form path on a fixed sparse `jac_prototype`: warmed `step!`s that form a
# new Jacobian update `JᵀJ` in place.

function bratu!(r, u, p)
    n = length(u)
    invh2 = p.invh2
    λ = p.λ
    @inbounds begin
        r[1] = (-2u[1] + u[2]) * invh2 + λ * exp(u[1])
        for i in 2:(n - 1)
            r[i] = (u[i - 1] - 2u[i] + u[i + 1]) * invh2 + λ * exp(u[i])
        end
        r[n] = (u[n - 1] - 2u[n]) * invh2 + λ * exp(u[n])
    end
    return nothing
end

function make_sparse_bratu(n = 1000; λ = 1.0)
    h = 1 / (n + 1)
    invh2 = 1 / h^2
    jac_prototype = spdiagm(
        -1 => fill(invh2, n - 1), 0 => fill(-2 * invh2, n), 1 => fill(invh2, n - 1)
    )
    colorvec = column_colors(coloring(jac_prototype, ColoringProblem(), GreedyColoringAlgorithm()))
    f = NonlinearFunction(bratu!; jac_prototype, colorvec)
    return NonlinearProblem(f, zeros(n), (; invh2, λ))
end

@testset "TrustRegion sparse step! allocations" begin
    prob = make_sparse_bratu()
    alg = TrustRegion()

    cache = init(prob, alg; maxiters = 100)
    for _ in 1:3
        step!(cache)
    end

    njacs = cache.stats.njacs
    allocs = ntuple(_ -> (@allocated step!(cache)), 5)
    @test cache.stats.njacs > njacs
    @test maximum(allocs) < 1000

    sol = solve(prob, alg; maxiters = 100)
    @test norm(sol.resid) < 1.0e-8
end

@testset "TrustRegion sparse reinit! with new p (linsolve = $(linsolve))" for linsolve in (
        nothing, KLUFactorization(),
    )
    prob = make_sparse_bratu(200)
    alg = TrustRegion(; linsolve)
    p₂ = (; prob.p..., λ = 3.0)

    cache = init(prob, alg; maxiters = 100)
    solve!(cache)
    reinit!(cache, prob.u0; p = p₂)
    sol₂ = solve!(cache)
    fresh = solve(remake(prob; p = p₂), alg; maxiters = 100)

    @test norm(sol₂.resid) < 1.0e-8
    @test sol₂.retcode == fresh.retcode
    @test sol₂.u == fresh.u
    @test cache.jac_cache.J isa SparseMatrixCSC
    @test cache.descent_cache.JᵀJ isa SparseMatrixCSC
    @test cache.descent_cache.JᵀJ == transpose(cache.jac_cache.J) * cache.jac_cache.J
end

# In-place `jac!` that inserts new structural entries into `jac_prototype` must
# still form a correct `JᵀJ` (pattern-mismatch fallback), matching dense TrustRegion.

f_grow(u, p) = [
    u[1] + 0.5 * u[2]^2 - 1.0,
    u[2] + 0.5 * u[1] * u[3] - 2.0,
    u[3] + 0.5 * u[1]^2 - 3.0,
]
j_grow(u, p) = dropzeros!(
    sparse(
        [
            1.0 u[2] 0.0
            0.5 * u[3] 1.0 0.5 * u[1]
            u[1] 0.0 1.0
        ]
    )
)
function f_grow!(r, u, p)
    r .= f_grow(u, p)
    return nothing
end
function j_grow!(J, u, p)
    Jn = j_grow(u, p)
    fill!(nonzeros(J), 0)
    for (i, j, v) in zip(findnz(Jn)...)
        J[i, j] = v
    end
    return nothing
end

@testset "TrustRegion growing sparse jac! matches dense" begin
    u0 = zeros(3)
    sparse_prob = NonlinearProblem(
        NonlinearFunction(f_grow!; jac = j_grow!, jac_prototype = sparse(1.0I, 3, 3)), u0
    )
    dense_prob = NonlinearProblem(NonlinearFunction(f_grow; jac = j_grow), u0)

    sparse_sol = solve(sparse_prob, TrustRegion(); maxiters = 200, abstol = 1.0e-10)
    dense_sol = solve(dense_prob, TrustRegion(); maxiters = 200, abstol = 1.0e-10)

    @test sparse_sol.retcode == dense_sol.retcode
    @test sparse_sol.u ≈ dense_sol.u rtol = 1.0e-10 atol = 1.0e-10
    @test norm(sparse_sol.resid) ≈ norm(dense_sol.resid) rtol = 1.0e-10 atol = 1.0e-10

    cache = init(sparse_prob, TrustRegion(); maxiters = 50)
    for _ in 1:4
        step!(cache)
    end
    J = cache.jac_cache.J
    JᵀJ = cache.descent_cache.JᵀJ
    @test J isa SparseMatrixCSC
    @test JᵀJ isa SparseMatrixCSC
    @test maximum(abs, Matrix(JᵀJ) - Matrix(transpose(J) * J)) == 0
end
