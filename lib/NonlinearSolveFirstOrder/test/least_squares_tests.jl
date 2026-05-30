@testsetup module CoreNLLSTesting

using NonlinearSolveFirstOrder, LineSearch, ADTypes, LinearSolve
using LineSearches: LineSearches
using ForwardDiff, FiniteDiff, Zygote

include("../../../common/common_nlls_testing.jl")

linesearches = []
for lsmethod in [
        LineSearches.Static(), LineSearches.HagerZhang(), LineSearches.MoreThuente(),
        LineSearches.StrongWolfe(), LineSearches.BackTracking(),
    ]
    push!(linesearches, LineSearchesJL(; method = lsmethod))
end
push!(linesearches, BackTracking())

solvers = []
for linsolve in [nothing, LUFactorization(), KrylovJL_GMRES(), KrylovJL_LSMR()]
    vjp_autodiffs = linsolve isa KrylovJL ? [nothing, AutoZygote(), AutoFiniteDiff()] :
        [nothing]
    for linesearch in linesearches, vjp_autodiff in vjp_autodiffs

        push!(solvers, GaussNewton(; linsolve, linesearch, vjp_autodiff))
    end
end
append!(
    solvers,
    [
        LevenbergMarquardt(),
        LevenbergMarquardt(; linsolve = LUFactorization()),
        LevenbergMarquardt(; linsolve = KrylovJL_GMRES()),
        LevenbergMarquardt(; linsolve = KrylovJL_LSMR()),
    ]
)
for radius_update_scheme in [
        RadiusUpdateSchemes.Simple, RadiusUpdateSchemes.NocedalWright,
        RadiusUpdateSchemes.NLsolve, RadiusUpdateSchemes.Hei,
        RadiusUpdateSchemes.Yuan, RadiusUpdateSchemes.Fan, RadiusUpdateSchemes.Bastin,
    ]
    push!(solvers, TrustRegion(; radius_update_scheme))
end

export solvers

end

@testitem "General NLLS Solvers" setup = [CoreNLLSTesting] tags = [:core] begin
    using LinearAlgebra

    nlls_problems = [prob_oop, prob_iip, prob_oop_vjp, prob_iip_vjp]

    for prob in nlls_problems, solver in solvers

        sol = solve(prob, solver; maxiters = 10000, abstol = 1.0e-6)
        @test SciMLBase.successful_retcode(sol)
        @test norm(sol.resid, 2) < 1.0e-6
    end
end

@testitem "Overdetermined NLLS with StaticArrays" tags = [:core] begin
    using NonlinearSolveFirstOrder, LinearAlgebra, StaticArrays

    # Overdetermined: 4 residuals, 2 unknowns (issue #560)
    function f_static(u, p)
        x, y = u
        return SA[
            sin(x) + y^2 - 1.0,
            x^2 + y - 2.0,
            exp(x) + x * y - 3.0,
            x + y - 1.0,
        ]
    end
    f_vec(u, p) = collect(f_static(u, p))

    prob_static = NonlinearLeastSquaresProblem(NonlinearFunction(f_static), SA[0.0, 0.0])
    prob_vec = NonlinearLeastSquaresProblem(NonlinearFunction(f_vec), [0.0, 0.0])

    for solver in (GaussNewton(), LevenbergMarquardt(), TrustRegion())
        sol_static = solve(prob_static, solver; maxiters = 10000, abstol = 1.0e-6)
        sol_vec = solve(prob_vec, solver; maxiters = 10000, abstol = 1.0e-6)
        @test SciMLBase.successful_retcode(sol_static)

        # We don't test the residuals themselves because they're quite large for
        # an overdetermined system. Instead we just check the results between
        # Vector/StaticArray match.
        @test norm(collect(sol_static.u) - sol_vec.u) < 1.0e-6
    end
end

@testitem "NLSS using GaussNewton for sparse matrices" tags = [:core] begin
    using NonlinearSolveFirstOrder, LinearAlgebra, SparseArrays, SciMLBase

    # Overdetermined: 4 residuals, 3 unknowns.
    # Has an exact solution at u = [1.0, 1.0, 1.0].
    function f!(r, u, p)
        r[1] = u[1]^2 + u[2]^2 - 2.0
        r[2] = u[2]^2 + u[3]^2 - 2.0
        r[3] = u[1] * u[2] - 1.0
        r[4] = u[2] * u[3] - 1.0
    end

    function jac!(J, u, p)
        fill!(J, 0)
        J[1, 1] = 2 * u[1];  J[1, 2] = 2 * u[2]
        J[2, 2] = 2 * u[2];  J[2, 3] = 2 * u[3]
        J[3, 1] = u[2];      J[3, 2] = u[1]
        J[4, 2] = u[3];      J[4, 3] = u[2]
    end

    proto = sparse([1, 1, 2, 2, 3, 3, 4, 4], [1, 2, 2, 3, 1, 2, 2, 3], ones(8), 4, 3)

    prob_sparse = NonlinearLeastSquaresProblem(
        NonlinearFunction(f!;
            jac = jac!,
            resid_prototype = zeros(4),
            jac_prototype = proto,
        ),
        [0.5, 0.5, 0.5],
    )
    prob_dense = NonlinearLeastSquaresProblem(
        NonlinearFunction(f!;
            jac = jac!,
            resid_prototype = zeros(4),
        ),
        [0.5, 0.5, 0.5],
    )

    sol_sparse = solve(prob_sparse, GaussNewton(); maxiters = 100, abstol = 1.0e-8)
    sol_dense  = solve(prob_dense,  GaussNewton(); maxiters = 100, abstol = 1.0e-8)

    @test SciMLBase.successful_retcode(sol_sparse)
    @test SciMLBase.successful_retcode(sol_dense)
    # Sparse and dense analytical Jacobians must converge to the same solution
    @test norm(sol_sparse.u - sol_dense.u) < 1.0e-6
end