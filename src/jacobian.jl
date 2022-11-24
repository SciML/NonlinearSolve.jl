mutable struct JacobianWrapper{fType, pType}
    f::fType
    p::pType
end

(uf::JacobianWrapper)(u) = uf.f(u, uf.p)
(uf::JacobianWrapper)(res, u) = uf.f(res, u, uf.p)

struct ImmutableJacobianWrapper{fType, pType}
    f::fType
    p::pType
end

(uf::ImmutableJacobianWrapper)(u) = uf.f(u, uf.p)

function calc_J(solver, cache)
    @unpack u, f, p, alg = solver
    @unpack uf = cache
    uf.f = f
    uf.p = p
    J = jacobian(uf, u, solver)
    return J
end

function calc_J(solver, uf::ImmutableJacobianWrapper)
    @unpack u, f, p, alg = solver
    J = jacobian(uf, u, solver)
    return J
end

function jacobian(f, x::Number, solver)
    if alg_autodiff(solver.alg)
        J = ForwardDiff.derivative(f, x)
    else
        J = FiniteDiff.finite_difference_derivative(f, x, alg_difftype(solver.alg),
                                                    eltype(x))
    end
    return J
end

function jacobian(f, x, solver)
    if alg_autodiff(solver.alg)
        J = ForwardDiff.jacobian(f, x)
    else
        J = FiniteDiff.finite_difference_jacobian(f, x, alg_difftype(solver.alg), eltype(x))
    end
    return J
end

function calc_J!(J, solver, cache)
    @unpack f, u, p, alg = solver
    @unpack du1, uf, jac_config = cache

    uf.f = f
    uf.p = p

    jacobian!(J, uf, u, du1, solver, jac_config)
end

function jacobian!(J, f, x, fx, solver, jac_config)
    alg = solver.alg
    if alg_autodiff(alg)
        ForwardDiff.jacobian!(J, f, fx, x, jac_config)
    else
        FiniteDiff.finite_difference_jacobian!(J, f, x, jac_config)
    end
    nothing
end
