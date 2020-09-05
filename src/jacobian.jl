mutable struct JacobianWrapper{fType, pType}
    f::fType
    p::pType
end

(uf::JacobianWrapper)(u) = uf.f(u, uf.p)
(uf::JacobianWrapper)(res, u) = uf.f(res, u, uf.p)

function calc_J(solver, cache)
    @unpack u, f, p, alg = solver
    @unpack uf = cache
    uf.f = f
    uf.p = p
    J = jacobian(uf, u, solver)
    return J
end

function jacobian(f, x, solver)
    if alg_autodiff(solver.alg)
        J = ForwardDiff.derivative(f, x)
    else
        J = FiniteDiff.finite_difference_derivative(f, x, solver.alg.diff_type, eltype(x))
    end
    return J
end
