using NonlinearSolveFirstOrder

using LinearAlgebra, SparseArrays, SparseConnectivityTracer, ADTypes,
    SparseMatrixColorings
using DifferentiationInterface: DenseSparsityDetector

const N = 32
const xyd_brusselator = range(0, stop = 1, length = N)

brusselator_f(x, y) = (((x - 0.3)^2 + (y - 0.6)^2) <= 0.1^2) * 5.0
limit(a, N) = a == N + 1 ? 1 : a == 0 ? N : a

function brusselator_2d_loop(du, u, p)
    A, B, alpha, dx = p
    alpha = alpha / dx^2
    return @inbounds for I in CartesianIndices((N, N))
        i, j = Tuple(I)
        x, y = xyd_brusselator[I[1]], xyd_brusselator[I[2]]
        ip1, im1,
            jp1, jm1 = limit(i + 1, N), limit(i - 1, N), limit(j + 1, N),
            limit(j - 1, N)
        du[i, j, 1] = alpha *
            (
            u[im1, j, 1] + u[ip1, j, 1] + u[i, jp1, 1] + u[i, jm1, 1] -
                4u[i, j, 1]
        ) +
            B +
            u[i, j, 1]^2 * u[i, j, 2] - (A + 1) * u[i, j, 1] +
            brusselator_f(x, y)
        du[i, j, 2] = alpha *
            (
            u[im1, j, 2] + u[ip1, j, 2] + u[i, jp1, 2] + u[i, jm1, 2] -
                4u[i, j, 2]
        ) + A * u[i, j, 1] - u[i, j, 1]^2 * u[i, j, 2]
    end
end

p = (3.4, 1.0, 10.0, step(xyd_brusselator))

function init_brusselator_2d(xyd)
    N = length(xyd)
    u = zeros(N, N, 2)
    for I in CartesianIndices((N, N))
        x = xyd[I[1]]
        y = xyd[I[2]]
        u[I, 1] = 22 * (y * (1 - y))^(3 / 2)
        u[I, 2] = 27 * (x * (1 - x))^(3 / 2)
    end
    return u
end

u0 = init_brusselator_2d(xyd_brusselator)
prob_brusselator_2d = NonlinearProblem(brusselator_2d_loop, u0, p)
sol = solve(prob_brusselator_2d, NewtonRaphson(); abstol = 1.0e-8)
@test norm(sol.resid, Inf) < 1.0e-8

prob_brusselator_2d_sparse = NonlinearProblem(
    NonlinearFunction(brusselator_2d_loop; sparsity = TracerSparsityDetector()),
    u0, p
)
sol = solve(prob_brusselator_2d_sparse, NewtonRaphson(); abstol = 1.0e-8)
@test norm(sol.resid, Inf) < 1.0e-8

# AD-based sparsity detection (DenseSparsityDetector) differentiates the function with its
# own tag. Combined with the default AutoSpecialize FunctionWrapper this used to throw
# "No matching function wrapper was found!"; the Jacobian cache must unwrap to the raw
# function for the detector. (SparseMatrixColorings is loaded above, so the sparse
# coloring path is active.)
prob_brusselator_2d_dense_di = NonlinearProblem(
    NonlinearFunction(
        brusselator_2d_loop;
        sparsity = DenseSparsityDetector(AutoForwardDiff(); atol = 1.0e-4)
    ),
    u0, p
)
sol = solve(prob_brusselator_2d_dense_di, NewtonRaphson(); abstol = 1.0e-8)
@test norm(sol.resid, Inf) < 1.0e-8

f! = (du, u) -> brusselator_2d_loop(du, u, p)
du0 = similar(u0)
jac_prototype = ADTypes.jacobian_sparsity(f!, du0, u0, TracerSparsityDetector())

ff_iip = NonlinearFunction(brusselator_2d_loop; jac_prototype)
prob_brusselator_2d = NonlinearProblem(ff_iip, u0, p)

sol = solve(prob_brusselator_2d, NewtonRaphson(); abstol = 1.0e-8)
@test norm(sol.resid, Inf) < 1.0e-8

sol = solve(
    prob_brusselator_2d,
    NewtonRaphson(autodiff = AutoFiniteDiff()); abstol = 1.0e-8
)
@test norm(sol.resid, Inf) < 1.0e-8
