using NonlinearSolveFirstOrder
using ADTypes: AutoForwardDiff
using ForwardDiff
using LinearAlgebra
using StaticArrays
using LinearSolve
const LS = LinearSolve

function f(u, p)
    L, U = cholesky(p.Σ)
    rhs = (u .* u .- p.λ)
    # there are some issues currently with LinearSolve and triangular matrices,
    # so we just make `L` dense here.
    linprob = LinearProblem(Matrix(L), rhs)
    alg = LS.GenericLUFactorization()
    sol = LinearSolve.solve(linprob, alg)
    return sol.u
end

struct MyParams{T, M}
    λ::T
    Σ::M
end

const autodiff = AutoForwardDiff(; chunksize = 1)
const alg = TrustRegion(; autodiff, linsolve = LS.CholeskyFactorization())
const prob = NonlinearLeastSquaresProblem{false}(f, rand(2), MyParams(rand(), hermitianpart(rand(2, 2) + 2I)))
const cache = init(prob, alg)

function minimize(x)
    ps = MyParams(x, hermitianpart(rand(2, 2) + 2I))
    reinit!(cache, rand(2); p = ps)
    solve!(cache)
    return cache
end
