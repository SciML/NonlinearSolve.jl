using NonlinearSolve

using StaticArrays, PolyesterForwardDiff

f_oop(u, p) = u .* u .- p

N = 4
u0 = SVector{N, Float64}(ones(N) .+ randn(N) * 0.01)

nlprob = NonlinearProblem(f_oop, u0, 2.0)

@test !(solve(nlprob, NewtonRaphson()).alg.autodiff isa AutoPolyesterForwardDiff)
