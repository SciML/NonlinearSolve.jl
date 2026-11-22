using NonlinearSolveFirstOrder, NonlinearSolveBase, LinearSolve
using LinearAlgebra, SparseArrays, SparseMatrixColorings, StableRNGs

const Utils = NonlinearSolveBase.Utils

@testset "normal_form_jacobian!! matches transpose(J) * J bitwise" begin
    rng = StableRNG(0)
    @testset "size $(m)×$(n)" for (m, n) in ((40, 40), (60, 25), (25, 60))
        J = sprandn(rng, m, n, 0.2)
        ws = Utils.normal_form_workspace(J)
        JᵀJ = transpose(J) * J
        nonzeros(J) .= randn(rng, nnz(J))
        expected = transpose(J) * J
        @test Utils.normal_form_jacobian!!(JᵀJ, J, ws) === JᵀJ
        @test JᵀJ.colptr == expected.colptr && JᵀJ.rowval == expected.rowval
        @test reinterpret(UInt64, nonzeros(JᵀJ)) == reinterpret(UInt64, nonzeros(expected))

        # A Jacobian whose pattern no longer matches the cache takes the allocating path.
        J₂ = sprandn(rng, m, n, 0.3)
        JᵀJ₂ = Utils.normal_form_jacobian!!(JᵀJ, J₂, ws)
        @test JᵀJ₂ == transpose(J₂) * J₂
    end
end

function bratu!(r, u, p)
    n = length(u)
    r[1] = (-2u[1] + u[2]) * p.invh2 + p.λ * exp(u[1])
    for i in 2:(n - 1)
        r[i] = (u[i - 1] - 2u[i] + u[i + 1]) * p.invh2 + p.λ * exp(u[i])
    end
    r[n] = (u[n - 1] - 2u[n]) * p.invh2 + p.λ * exp(u[n])
    return nothing
end

@testset "step! does not allocate (disable_geodesic = $(geo))" for geo in (Val(false), Val(true))
    n = 200
    invh2 = (n + 1)^2
    jac_prototype = spdiagm(-1 => ones(n - 1), 0 => ones(n), 1 => ones(n - 1))
    colorvec = column_colors(
        coloring(jac_prototype, ColoringProblem(), GreedyColoringAlgorithm())
    )
    f = NonlinearFunction(bratu!; jac_prototype, colorvec)
    prob = NonlinearProblem(f, zeros(n), (; invh2, λ = 1.0))
    alg = LevenbergMarquardt(; linsolve = KLUFactorization(), disable_geodesic = geo)

    cache = init(prob, alg; maxiters = 1000, abstol = 0.0, reltol = 0.0)
    for _ in 1:3
        step!(cache)
    end
    @test maximum(_ -> (@allocated step!(cache)), 1:5) < 1000
end
