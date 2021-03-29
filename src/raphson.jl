struct NewtonRaphson{CS, AD, DT, L} <: AbstractNewtonAlgorithm{CS,AD}
    diff_type::DT
    linsolve::L
end

function NewtonRaphson(;autodiff=true,chunk_size=12,diff_type=Val{:forward},linsolve=DEFAULT_LINSOLVE)
    NewtonRaphson{chunk_size, autodiff, typeof(diff_type), typeof(linsolve)}(diff_type, linsolve)
end

mutable struct NewtonRaphsonCache{ufType, L, jType, uType, JC}
    uf::ufType
    linsolve::L
    J::jType
    du1::uType
    jac_config::JC
end

function alg_cache(alg::NewtonRaphson, f, u, p, ::Val{true})
    uf = JacobianWrapper(f,p)
    linsolve = alg.linsolve(Val{:init}, f, u)
    J = false .* u .* u'
    du1 = zero(u)
    tmp = zero(u)
    if alg_autodiff(alg)
        jac_config = ForwardDiff.JacobianConfig(uf, du1, u)
    else
        if alg.diff_type != Val{:complex}
            du2 = zero(u)
            jac_config = FiniteDiff.JacobianCache(tmp, du1, du2, alg.diff_type)
        else
            jac_config = FiniteDiff.JacobianCache(Complex{eltype(tmp)}.(tmp),Complex{eltype(du1)}.(du1),nothing,alg.diff_type,eltype(u))
        end
    end
    NewtonRaphsonCache(uf, linsolve, J, du1, jac_config)
end

function alg_cache(alg::NewtonRaphson, f, u, p, ::Val{false})
    nothing
end

function perform_step(solver::NewtonImmutableSolver, alg::NewtonRaphson, ::Val{true})
    @unpack u, fu, f, p, cache = solver
    @unpack J, linsolve, du1 = cache
    calc_J!(J, solver, cache)
    # u = u - J \ fu
    linsolve(du1, J, fu, true)
    @. u = u - du1
    f(fu, u, p)
    if solver.internalnorm(solver.fu) < solver.tol
        @set! solver.force_stop = true
    end
    return solver
end

function perform_step(solver::NewtonImmutableSolver, alg::NewtonRaphson, ::Val{false})
    @unpack u, fu, f, p = solver
    J = calc_J(solver, ImmutableJacobianWrapper(f, p))
    @set! solver.u = u - J \ fu
    fu = f(solver.u, p)
    @set! solver.fu = fu
    if iszero(solver.fu) || solver.internalnorm(solver.fu) < solver.tol
        @set! solver.force_stop = true
    end
    return solver
end
