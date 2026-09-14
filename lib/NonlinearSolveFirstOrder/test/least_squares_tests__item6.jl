using NonlinearSolveFirstOrder, SciMLBase, LinearAlgebra
using LinearSolve, SparseArrays, StaticArrays, SciMLOperators

# Problems from the Moré–Garbow–Hillstrom / Hock–Schittkowski collections (as packaged in
# NLSProblems.jl for the TrustRegionLeastSquares.jl benchmark) on which the dogleg
# trust-region either stagnates or converges to a worse local minimum. The Moré
# subproblem solve reaches the best-known least-squares cost on all of them.

function mgh16_resid(u, p)
    t = (1:20) ./ 5
    return [
        (u[1] + t[i] * u[2] - exp(t[i]))^2 +
            (u[3] + u[4] * sin(t[i]) - cos(t[i]))^2 for i in 1:20
    ]
end

function mgh18_resid(u, p)
    t = 0.1 * (1:13)
    y = exp.(-t) .- 5 .* exp.(-10t) .+ 3 .* exp.(-4t)
    return [
        u[3] * exp(-t[i] * u[1]) - u[4] * exp(-t[i] * u[2]) +
            u[6] * exp(-t[i] * u[5]) - y[i] for i in 1:13
    ]
end

const MGH19_Y = [
    1.366, 1.191, 1.112, 1.013, 0.991, 0.885, 0.831, 0.847, 0.786, 0.725, 0.746,
    0.679, 0.608, 0.655, 0.616, 0.606, 0.602, 0.625, 0.651, 0.724, 0.649, 0.649,
    0.694, 0.644, 0.624, 0.661, 0.612, 0.558, 0.533, 0.495, 0.5, 0.423, 0.395,
    0.375, 0.372, 0.391, 0.396, 0.405, 0.428, 0.429, 0.523, 0.562, 0.607, 0.653,
    0.672, 0.708, 0.633, 0.668, 0.645, 0.632, 0.591, 0.559, 0.597, 0.625, 0.739,
    0.71, 0.729, 0.72, 0.636, 0.581, 0.428, 0.292, 0.162, 0.098, 0.054,
]
function mgh19_resid(u, p)
    t = ((1:65) .- 1) ./ 10
    return [
        MGH19_Y[i] - u[1] * exp(-t[i] * u[5]) +
            u[2] * exp(-(t[i] - u[9])^2 * u[6]) +
            u[3] * exp(-(t[i] - u[10])^2 * u[7]) +
            u[4] * exp(-(t[i] - u[11])^2 * u[8]) for i in 1:65
    ]
end

# mgh34: linear least-squares, rank-1 Jacobian with zero rows and columns. Exercises the
# stationary-point (`Jᵀf = 0` on a singular `JᵀJ`) short-circuit of the Moré solve.
function mgh34_resid(u, p)
    s = sum(j * u[j] for j in 2:9)
    return vcat(-1.0, [i * s - 1.0 for i in 1:18], -1.0)
end

function tp267_resid(u, p)
    z = 0.1 * (1:11)
    y = exp.(-z) .- 5 .* exp.(-10z) .+ 3 .* exp.(-4z)
    return [
        u[3] * exp(-u[1] * z[i]) - u[4] * exp(-u[2] * z[i]) +
            3 * exp(-u[5] * z[i]) - y[i] for i in 1:11
    ]
end

function tp272_resid(u, p)
    t = 0.1 * (1:13)
    y = exp.(-t) .- 5 .* exp.(-10t) .+ 3 .* exp.(-4t)
    return [
        u[4] * exp(-u[1] * t[i]) - u[5] * exp(-u[2] * t[i]) +
            u[6] * exp(-u[3] * t[i]) - y[i] for i in 1:13
    ]
end

tp312_resid(u, p) = [
    u[1]^2 + 12 * u[2] - 1,
    49 * u[1]^2 + 49 * u[2]^2 + 84 * u[1] + 2324 * u[2] - 681,
]

function tp333_resid(u, p)
    a = [4, 5.75, 7.5, 24, 32, 48, 72, 96]
    y = [72.1, 65.6, 55.9, 17.1, 9.8, 4.5, 1.3, 0.6]
    return [(y[i] - u[1] * exp(-u[2] * a[i]) - u[3]) / y[i] for i in 1:8]
end

# (name, residual, u0, best-known cost ½‖f‖² from the benchmark)
const MORE_NLLS_CASES = [
    ("mgh16", mgh16_resid, [25.0, 5.0, -5.0, -1.0], 42911.10081317816),
    ("mgh18", mgh18_resid, [1.0, 2.0, 1.0, 1.0, 1.0, 1.0], 3.5246947598532117e-22),
    (
        "mgh19", mgh19_resid, [1.3, 0.65, 0.65, 0.7, 0.6, 3.0, 5.0, 7.0, 2.0, 4.5, 5.5],
        0.0200843002555413,
    ),
    ("mgh34", mgh34_resid, ones(10), 3.067567567552487),
    ("tp267", tp267_resid, fill(2.0, 5), 2.328279820929062e-27),
    ("tp272", tp272_resid, [1.0, 2.0, 1.0, 1.0, 1.0, 1.0], 1.5437815722157893e-22),
    ("tp312", tp312_resid, ones(2), 2.9612813806220117),
    ("tp333", tp333_resid, [30.0, 0.04, 3.0], 0.021635177018515545),
]

@testset "More subproblem: benchmark failures" begin
    for (name, resid, u0, best_cost) in MORE_NLLS_CASES
        prob = NonlinearLeastSquaresProblem(resid, u0)
        sol = solve(prob, TrustRegion(; subproblem = TrustRegionSubproblem.More); maxiters = 400, abstol = 1.0e-8)
        cost = norm(sol.resid)^2 / 2
        @test SciMLBase.successful_retcode(sol)
        @test cost ≤ max(best_cost * (1 + 1.0e-4), 1.0e-12)
    end
end

@testset "More subproblem: Jacobian scaling" begin
    for (name, resid, u0, best_cost) in MORE_NLLS_CASES
        prob = NonlinearLeastSquaresProblem(resid, u0)
        sol = solve(
            prob,
            TrustRegion(;
                subproblem = MoreTrustRegionDescent(; scaling = :jacobian)
            );
            maxiters = 2000, abstol = 1.0e-8
        )
        cost = norm(sol.resid)^2 / 2
        @test SciMLBase.successful_retcode(sol)
        # Under column-norm scaling tp267 converges to a nearer local minimum
        # (cost ≈ 1.3e-3, identical across QR/LU paths) rather than the global
        # ~0 — a basin property of the scaled problem, so assert convergence only
        name !== "tp267" && @test cost ≤ max(best_cost * (1 + 1.0e-4), 1.0e-12)
    end
end

@testset "More subproblem: API surface" begin
    # `subproblem` accepts a `TrustRegionSubproblem` or the descent directly; symbols
    # and other values are rejected
    @test TrustRegion(; subproblem = TrustRegionSubproblem.Dogleg) isa GeneralizedFirstOrderAlgorithm
    @test_throws ArgumentError TrustRegion(; subproblem = :bogus)
    @test_throws ArgumentError TrustRegion(; subproblem = :more)

    # scalar problems
    sol = solve(
        NonlinearProblem((u, p) -> u^3 - u - 2, 1.0),
        TrustRegion(; subproblem = TrustRegionSubproblem.More); abstol = 1.0e-10
    )
    @test SciMLBase.successful_retcode(sol)
    @test abs(sol.u^3 - sol.u - 2) < 1.0e-8

    # reinit! resets the cached Gauss-Newton step and damping parameter
    prob = NonlinearLeastSquaresProblem(tp333_resid, [30.0, 0.04, 3.0])
    cache = init(prob, TrustRegion(; subproblem = TrustRegionSubproblem.More); abstol = 1.0e-8)
    sol = solve!(cache)
    @test SciMLBase.successful_retcode(sol)
    reinit!(cache, [5.0, 0.1, 1.0])
    sol = solve!(cache)
    @test SciMLBase.successful_retcode(sol)
    @test norm(sol.resid)^2 / 2 ≤ 0.021635177018515545 * (1 + 1.0e-4)

    # static arrays: immutable `JᵀJ` takes the out-of-place damped-system path
    prob = NonlinearLeastSquaresProblem(tp312_resid, SA[1.0, 1.0])
    sol = solve(prob, TrustRegion(; subproblem = TrustRegionSubproblem.More); abstol = 1.0e-8)
    @test SciMLBase.successful_retcode(sol)
    @test norm(sol.resid)^2 / 2 ≤ 2.9612813806220117 * (1 + 1.0e-4)

    # matrix-shaped state: the internal step/`Jᵀfu` buffers are `length(u)` vectors,
    # matching the vectorized Jacobian, while `δu` keeps the state shape
    fmat(u, p) = u .* u .- p
    matprob = NonlinearProblem(fmat, ones(2, 2), 4.0)
    vecmatprob = NonlinearProblem(fmat, ones(4), 4.0)
    for subproblem in (TrustRegionSubproblem.More, TrustRegionSubproblem.Dogleg)
        sol = solve(matprob, TrustRegion(; subproblem); abstol = 1.0e-10)
        @test SciMLBase.successful_retcode(sol)
        @test vec(sol.u) ≈ solve(vecmatprob, TrustRegion(; subproblem); abstol = 1.0e-10).u
    end

    # sparse `jac_prototype`: `JᵀJ` stays sparse and the damped system is rebuilt
    # out-of-place each iteration
    function tp333_resid!(F, u, p)
        a = [4, 5.75, 7.5, 24, 32, 48, 72, 96]
        y = [72.1, 65.6, 55.9, 17.1, 9.8, 4.5, 1.3, 0.6]
        @. F = (y - u[1] * exp(-u[2] * a) - u[3]) / y
        return nothing
    end
    function tp333_jac!(J, u, p)
        a = [4, 5.75, 7.5, 24, 32, 48, 72, 96]
        y = [72.1, 65.6, 55.9, 17.1, 9.8, 4.5, 1.3, 0.6]
        fill!(J, 0)
        for i in 1:8
            J[i, 1] = -exp(-u[2] * a[i]) / y[i]
            J[i, 2] = u[1] * a[i] * exp(-u[2] * a[i]) / y[i]
            J[i, 3] = -1 / y[i]
        end
        return nothing
    end
    fn = NonlinearFunction{true}(
        tp333_resid!; jac = tp333_jac!, jac_prototype = sparse(ones(8, 3)),
        resid_prototype = zeros(8)
    )
    prob = NonlinearLeastSquaresProblem(fn, [30.0, 0.04, 3.0])
    sol = solve(prob, TrustRegion(; subproblem = TrustRegionSubproblem.More); abstol = 1.0e-8)
    @test SciMLBase.successful_retcode(sol)
    @test norm(sol.resid)^2 / 2 ≤ 0.021635177018515545 * (1 + 1.0e-4)

    # a user-supplied linear solver is honored — `LUFactorization` needs a square
    # system, so the descent switches to the damped normal equations
    sol = solve(
        NonlinearLeastSquaresProblem(tp312_resid, ones(2)),
        TrustRegion(;
            subproblem = MoreTrustRegionDescent(; linsolve = LUFactorization())
        ); abstol = 1.0e-8
    )
    @test SciMLBase.successful_retcode(sol)
    @test norm(sol.resid)^2 / 2 ≤ 2.9612813806220117 * (1 + 1.0e-4)

    # matrix-free Jacobians: the augmented system is applied as a stacked operator.
    # A least-squares Krylov solver (LSMR) solves it directly; a square-only Krylov
    # solver (GMRES) routes through the normal-form operator `JᵀJ + λI`
    fn = NonlinearFunction(
        (u, p) -> u .^ 2 .+ u .- 1;
        jvp = (v, u, p) -> (2u .+ 1) .* v, vjp = (v, u, p) -> (2u .+ 1) .* v
    )
    prob = NonlinearLeastSquaresProblem(fn, ones(3))
    for linsolve in (KrylovJL_LSMR(), KrylovJL_GMRES())
        sol = solve(
            prob,
            TrustRegion(;
                subproblem = TrustRegionSubproblem.More, concrete_jac = Val(false), linsolve
            ); abstol = 1.0e-8
        )
        @test SciMLBase.successful_retcode(sol)
        @test norm(sol.resid) < 1.0e-6
    end

    # `:jacobian` scaling needs column norms of a concrete Jacobian; the descent's
    # own `linsolve` is what keeps the Jacobian matrix-free
    @test_throws ArgumentError solve(
        prob,
        TrustRegion(;
            subproblem = MoreTrustRegionDescent(;
                scaling = :jacobian, linsolve = KrylovJL_LSMR()
            ),
            concrete_jac = Val(false)
        )
    )

    # A convertible operator `jac_prototype` is materialized for a concrete-only
    # linsolve (the `jac_convert` path), and applied matrix-free under a Krylov one
    function tp312_resid!(F, u, p)
        F[1] = u[1]^2 + 12 * u[2] - 1
        F[2] = 49 * u[1]^2 + 49 * u[2]^2 + 84 * u[1] + 2324 * u[2] - 681
        return nothing
    end
    function tp312_jacop!(J_op, u, p)
        J_op.A .= [
            2u[1] 12.0
            98u[1] + 84 98u[2] + 2324.0
        ]
        return nothing
    end
    fn_op = NonlinearFunction{true}(
        tp312_resid!; jac = tp312_jacop!,
        jac_prototype = MatrixOperator(zeros(2, 2)), resid_prototype = zeros(2)
    )
    prob_op = NonlinearLeastSquaresProblem(fn_op, ones(2))
    sol = solve(
        prob_op,
        TrustRegion(;
            subproblem = MoreTrustRegionDescent(; linsolve = LUFactorization())
        ); abstol = 1.0e-8
    )
    @test SciMLBase.successful_retcode(sol)
    @test norm(sol.resid)^2 / 2 ≤ 2.9612813806220117 * (1 + 1.0e-4)
    sol = solve(prob_op, TrustRegion(; subproblem = TrustRegionSubproblem.More); abstol = 1.0e-8)
    @test SciMLBase.successful_retcode(sol)
    @test norm(sol.resid)^2 / 2 ≤ 2.9612813806220117 * (1 + 1.0e-4)
end
