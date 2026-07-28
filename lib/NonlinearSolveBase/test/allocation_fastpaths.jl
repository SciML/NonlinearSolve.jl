using ADTypes: AutoForwardDiff
using ForwardDiff
using NonlinearSolveBase
using SciMLBase
using Test

function dense_jacobian_cache_allocations()
    function f!(du, u, p)
        du[1] = u[1]^2 + u[2] + p
        du[2] = u[1] + 3 * u[2] - p
        return nothing
    end

    u = [2.0, 3.0]
    p = 0.5
    f = NonlinearFunction{true, SciMLBase.FullSpecialize}(f!)
    prob = NonlinearProblem(f, u, p)
    fu = similar(u)
    f(fu, u, p)
    cache = NonlinearSolveBase.construct_jacobian_cache(
        prob, nothing, f, fu;
        stats = SciMLBase.NLStats(0, 0, 0, 0, 0),
        autodiff = AutoForwardDiff(; chunksize = 1),
    )

    cache(u)
    cache(u)
    J = copy(cache(u))
    allocs = @allocated cache(u)
    return J, allocs
end

J, jacobian_allocs = dense_jacobian_cache_allocations()
@test J == [4.0 1.0; 1.0 3.0]
@test jacobian_allocs == 0

function same_buffer_restructure_allocations()
    x = ones(4)
    NonlinearSolveBase.Utils.restructure(x, x)
    y = NonlinearSolveBase.Utils.restructure(x, x)
    allocs = @allocated NonlinearSolveBase.Utils.restructure(x, x)
    return x, y, allocs
end

x, y, restructure_allocs = same_buffer_restructure_allocations()
@test y === x
@test restructure_allocs == 0
