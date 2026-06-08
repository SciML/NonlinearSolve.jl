using NonlinearSolve

u0_broken = [rand(2), rand(2)]
f(u, p) = u
prob = NonlinearProblem(f, u0_broken)
@test_throws SciMLBase.NonNumberEltypeError solve(prob)
