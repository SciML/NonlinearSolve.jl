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

function minimize(x)
    autodiff = AutoForwardDiff(; chunksize=1)
    alg = TrustRegion(; autodiff, linsolve=LS.CholeskyFactorization())
    ps = MyParams(rand(), hermitianpart(rand(2,2)+2I))
    prob = NonlinearLeastSquaresProblem{false}(f, rand(2), ps)
    sol = solve(prob, alg)
    return sol
end
