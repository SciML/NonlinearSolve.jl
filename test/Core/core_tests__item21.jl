using NonlinearSolve

using LinearAlgebra
f(u, p) = -(u .- 0.1) .^ 3
prob = NonlinearProblem(f, [0.0, 0.0], 0)
mutable struct DummyPreconditioners
    i::Int
    reinit_check::Int
end
function (precs::DummyPreconditioners)(W, p = nothing)
    # Here we test that NonlinearSolve actually passes the parameters through
    # LinearSolve into the preconditioner constructor.
    @test p isa NonlinearSolveBase.LinearSolveParameters
    @test p.p == precs.reinit_check # p.p is the p of the nonlinear problem
    # By incrementing this variable we make sure that this function has been called at least once.
    precs.i += 1
    return LinearAlgebra.I, LinearAlgebra.I
end
# Check if it works in principle
precs = DummyPreconditioners(0, 0)
iter = init(prob, NewtonRaphson(linsolve = KrylovJL_GMRES(precs = precs), concrete_jac = false))
iinit = precs.i
solve!(iter)
@test precs.i > 0
iprev = precs.i
# Reinit with u0
precs.i = 0
precs.reinit_check = 1
reinit!(iter; u0 = [0.0, 0.0], p = precs.reinit_check)
ireinit = precs.i
solve!(iter)
@test precs.i - ireinit == iprev - iinit
# Reinit without passing u0
precs.i = 0
precs.reinit_check = 2
reinit!(iter; p = precs.reinit_check)
@test precs.i == 0
solve!(iter)
@test precs.i == 1
