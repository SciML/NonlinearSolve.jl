using JLArrays, NonlinearSolveFirstOrder, NonlinearSolveBase, LinearSolve, Logging
using ADTypes: AutoFiniteDiff, AutoForwardDiff, AutoPolyesterForwardDiff
using ForwardDiff: value
using SciMLBase: NLStats

JLArrays.allowscalar(false)
try
    jlarray_f(u, p) = u .^ 2 .- p
    jlarray_prob = NonlinearProblem(jlarray_f, JLArray([1.0, 2.0]), 4.0)

    jlarray_nr = solve(jlarray_prob, NewtonRaphson())
    @test jlarray_nr.retcode == ReturnCode.Success
    @test jlarray_nr.u isa JLArray
    @test Array(jlarray_nr.u) ≈ [2.0, 2.0]

    jlarray_tr = solve(jlarray_prob, TrustRegion())
    @test jlarray_tr.retcode == ReturnCode.Success
    @test jlarray_tr.u isa JLArray
    @test Array(jlarray_tr.u) ≈ [2.0, 2.0]

    # Matrix-free Krylov linsolve never materializes a concrete Jacobian, routing
    # through `JacobianOperator`'s eagerly-prepared VJP/JVP instead of `DI.jacobian`.
    # ForwardDiff pushforwards do not scalar-index, so they stay the JVP backend.
    jlarray_nr_krylov = solve(jlarray_prob, NewtonRaphson(linsolve = KrylovJL_GMRES()))
    @test jlarray_nr_krylov.retcode == ReturnCode.Success
    @test Array(jlarray_nr_krylov.u) ≈ [2.0, 2.0]
    @test jlarray_nr_krylov.alg.jvp_autodiff isa AutoForwardDiff

    @testset "Explicit ForwardDiff on the matrix-free path: $(kw)" for kw in (
            (; jvp_autodiff = AutoForwardDiff()), (; autodiff = AutoForwardDiff()),
        )
        alg = NewtonRaphson(; linsolve = KrylovJL_GMRES(), kw...)
        sol = @test_logs min_level = Logging.Warn solve(jlarray_prob, alg)
        @test sol.retcode == ReturnCode.Success
        @test Array(sol.u) ≈ [2.0, 2.0]
        @test sol.alg.jvp_autodiff isa AutoForwardDiff
    end

    jlarray_tr_krylov = solve(jlarray_prob, TrustRegion(linsolve = KrylovJL_GMRES()))
    @test jlarray_tr_krylov.retcode == ReturnCode.Success
    @test Array(jlarray_tr_krylov.u) ≈ [2.0, 2.0]

    # Yuan/Bastin radius updates eagerly compute `Jᵀfu` through the VJP operator at
    # `init` time, so they exercise the DI pullback path outside the Krylov cache too.
    jlarray_tr_yuan = solve(
        jlarray_prob, TrustRegion(radius_update_scheme = RadiusUpdateSchemes.Yuan)
    )
    @test jlarray_tr_yuan.retcode == ReturnCode.Success
    @test Array(jlarray_tr_yuan.u) ≈ [2.0, 2.0]

    jlarray_tr_bastin = solve(
        jlarray_prob, TrustRegion(radius_update_scheme = RadiusUpdateSchemes.Bastin)
    )
    @test jlarray_tr_bastin.retcode == ReturnCode.Success
    @test Array(jlarray_tr_bastin.u) ≈ [2.0, 2.0]

    @testset "Default Newton on a strided JLArray view" begin
        u0 = view(JLArray([1.0, 7.0, 2.0]), 1:2:3)
        prob = NonlinearProblem(jlarray_f, u0, 4.0)
        sol = solve(prob, NewtonRaphson())
        @test sol.retcode == ReturnCode.Success
        @test sol.u isa JLArray
        @test Array(sol.u) ≈ [2.0, 2.0]
    end

    @testset "Automatic Jacobian uses the working state" begin
        prob = NonlinearProblem(jlarray_f, [1.0, 2.0], 4.0)
        u = jlarray_prob.u0
        fu = jlarray_f(u, prob.p)
        jac_cache = NonlinearSolveBase.construct_jacobian_cache(
            prob, NewtonRaphson(), prob.f, fu, u; stats = NLStats(0, 0, 0, 0, 0)
        )
        J = jac_cache(u)
        @test J isa JLArray
        @test Array(J) ≈ [2.0 0.0; 0.0 4.0] rtol = 1.0e-6
    end
finally
    JLArrays.allowscalar(true)
end

using PolyesterForwardDiff: PolyesterForwardDiff

@testset "Explicit $(nameof(typeof(ad))) Jacobian on a scalar-indexable JLArray" for ad in (
        AutoForwardDiff(), AutoPolyesterForwardDiff(),
    )
    f(u, p) = u .^ 2 .- p
    u = JLArray([1.0, 2.0])
    prob = NonlinearProblem(f, u, 4.0)
    jac_cache = @test_logs min_level = Logging.Warn NonlinearSolveBase.construct_jacobian_cache(
        prob, NewtonRaphson(), prob.f, f(u, prob.p), u;
        stats = NLStats(0, 0, 0, 0, 0), autodiff = ad
    )
    J = jac_cache(u)
    @test J isa JLArray
    @test Array(J) == [2.0 0.0; 0.0 4.0]
end

function guarded_residual(u, evaluations)
    primal = value.(Array(u))
    push!(evaluations, primal)
    all(x -> 0.0 <= x <= 1.0 + 8eps(1.0), primal) || error("residual evaluated outside [0, 1]")
    return u .- 0.5
end

@testset "Explicit ForwardDiff Newton stays in the residual domain" begin
    prob = NonlinearProblem(guarded_residual, JLArray([1.0, 0.9]), Vector{Float64}[])
    sol = solve(prob, NewtonRaphson(autodiff = AutoForwardDiff()))
    @test sol.retcode == ReturnCode.Success
    @test sol.alg.autodiff isa AutoForwardDiff
    @test Array(sol.u) ≈ [0.5, 0.5]
end

@testset "Bounded trust region with autodiff = $(ad)" for ad in (
        nothing, AutoForwardDiff(), AutoFiniteDiff(),
    )
    evaluations = Vector{Float64}[]
    prob = NonlinearProblem(
        guarded_residual, JLArray([1.0, 0.9]), evaluations; lb = zeros(2), ub = ones(2)
    )
    sol = solve(prob, BoundedTrustRegion(; autodiff = ad))
    @test sol.retcode == ReturnCode.Success
    @test sol.alg.autodiff isa (ad isa AutoForwardDiff ? AutoForwardDiff : AutoFiniteDiff)
    @test Array(sol.u) ≈ [0.5, 0.5]
    @test !isempty(evaluations) && all(u -> all(x -> 0.0 <= x <= 1.0, u), evaluations)
end
