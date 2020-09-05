struct NewtonRaphson{CS, AD, DT} <: AbstractNewtonAlgorithm{CS,AD} 
    diff_type::DT
end

function NewtonRaphson(;autodiff=true,chunk_size=12,diff_type=Val{:forward})
    NewtonRaphson{chunk_size, autodiff, typeof(diff_type)}(diff_type)
end

mutable struct NewtonRaphsonCache{ufType}
    uf::ufType
end

function alg_cache(alg::NewtonRaphson, f, u, p, ::Val{true})
    uf = JacobianWrapper(f,p)
    NewtonRaphsonCache(uf)
end

function alg_cache(alg::NewtonRaphson, f, u, p, ::Val{false})
    uf = JacobianWrapper(f,p)
    NewtonRaphsonCache(uf)
end

function perform_step!(solver, alg::NewtonRaphson, cache)
    @unpack u, fu, f, p = solver
    J = calc_J(solver, cache)
    solver.u = u - J \ fu
    solver.fu = f(solver.u, p)
    if iszero(solver.fu) || abs(solver.fu) < solver.tol
        solver.force_stop = true
    end
end