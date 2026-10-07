using NonlinearSolveBase, SciMLBase, ADTypes, ForwardDiff, LinearAlgebra, Test

# The dual-number primal of `sin` is not bitwise equal to a plain `sin` call, so a Jacobian
# evaluation that wrote its f(u) into the residual would perturb it.
const N = 100
const A = [1 / (1 + abs(i - j)) for i in 1:N, j in 1:N] + 2N * I
f!(du, u, p) = (mul!(du, A, u); du .+= 1.5N .* sin.(u) .- p; nothing)

prob = NonlinearProblem(NonlinearFunction{true}(f!), fill(0.7, N), 1.0)
fu = similar(prob.u0)
f!(fu, prob.u0, prob.p)
jac_cache = NonlinearSolveBase.construct_jacobian_cache(
    prob, nothing, prob.f, fu; stats = SciMLBase.NLStats(0, 0, 0, 0, 0),
    autodiff = AutoForwardDiff(), linsolve = nothing
)
for u in (prob.u0, fill(-0.3, N))
    f!(fu, u, prob.p)
    fu_plain = copy(fu)
    J = jac_cache(u)
    @test fu == fu_plain
    @test J ≈ A + Diagonal(1.5N .* cos.(u))
end
