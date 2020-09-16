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

mutable struct NewtonRaphsonConstantCache{ufType}
    uf::ufType
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
    uf = JacobianWrapper(f,p)
    NewtonRaphsonConstantCache(uf)
end

function perform_step!(solver, alg::NewtonRaphson, cache::NewtonRaphsonConstantCache)
    @unpack u, fu, f, p = solver
    J = calc_J(solver, cache)
    solver.u = u - J \ fu
    solver.fu = f(solver.u, p)
    if iszero(solver.fu) || abs(solver.fu) < solver.tol
        solver.force_stop = true
    end
end

function perform_step!(solver, alg::NewtonRaphson, cache::NewtonRaphsonCache)
    @unpack u, fu, f, p = solver
    @unpack J, linsolve, du1 = cache
    calc_J!(J, solver, cache)
    # u = u - J \ fu
    linsolve(du1, J, fu, true)
    @. u = u - du1
    f(fu, u, p)
    if solver.internalnorm(solver.fu) < solver.tol
        solver.force_stop = true
    end
end
