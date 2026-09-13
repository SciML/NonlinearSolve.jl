using LineSearch: ArmijoLineSearch
using NonlinearSolveFirstOrder, NonlinearSolveBase, LinearSolve, SciMLOperators, SciMLBase, ADTypes, StaticArrays, SparseArrays, LinearAlgebra, ForwardDiff

# Dual comparisons include tangent components; feasibility concerns primal coordinates.
bounded_gn_tr(; kwargs...) = BoundedGaussNewton(; globalization = :trustregion, kwargs...)
bounded_gn_tr_ls(; kwargs...) = BoundedGaussNewton(; globalization = :trustregion_linesearch, kwargs...)
const native_bounded_algorithms = [
    TrustRegionReflective, BoundedLevenbergMarquardt, Dogbox,
    BoundedGaussNewton, bounded_gn_tr, bounded_gn_tr_ls,
]

@testset "Native bounded solver contracts" begin
    for constructor in native_bounded_algorithms, ad in (AutoForwardDiff(), AutoFiniteDiff())
        @testset "$(constructor), $(ad)" begin
            alg = constructor(; autodiff = ad)
            @test SciMLBase.allowsbounds(alg)
            for u0 in ([0.0, 0.0], [1.0, -2.0], [0.5, -1.0])
                evaluations = Ref(0)
                f = function (u, p)
                    @test all([0.0, -2.0] .<= ForwardDiff.value.(u) .<= [1.0, 0.0])
                    evaluations[] += 1
                    return [u[1] - 2, u[2] + 1, 1.0]
                end
                prob = NonlinearLeastSquaresProblem(f, u0; lb = [0.0, -2.0], ub = [1.0, 0.0])
                sol = solve(prob, alg; abstol = 1.0e-10, maxiters = 1000)
                @test SciMLBase.successful_retcode(sol)
                @test sol.u ≈ [1.0, -1.0] atol = 1.0e-6
                @test sol.resid ≈ [-1.0, 0.0, 1.0] atol = 1.0e-6
                @test evaluations[] > 0
            end
            for problem in (NonlinearProblem, NonlinearLeastSquaresProblem)
                sol = solve(problem((u, p) -> u - 2, 0.0; lb = 0.0, ub = 1.0), alg)
                @test SciMLBase.successful_retcode(sol) == (problem === NonlinearLeastSquaresProblem)
                @test sol.u ≈ 1.0 atol = 1.0e-6
                root = solve(problem((u, p) -> u - 1, 1.0; lb = 0.0, ub = 1.0), alg)
                @test SciMLBase.successful_retcode(root)
                @test abs(root.resid) < 1.0e-7
            end
            for fixed in (false, true)
                lb, ub = [0.0, -Inf], [fixed ? 0.0 : 1.0, Inf]
                f = (u, p) -> begin
                    @test all(lb .<= ForwardDiff.value.(u) .<= ub)
                    [u[1] - 2, u[2] - 3, 1.0]
                end
                sol = solve(NonlinearLeastSquaresProblem(f, [0.0, 0.0]; lb, ub), alg)
                @test SciMLBase.successful_retcode(sol)
                @test sol.u ≈ [ub[1], 3.0] atol = 1.0e-6
            end
            allfixed = NonlinearLeastSquaresProblem((u, p) -> u .- 2, [1.0, 1.0]; lb = 1.0, ub = 1.0)
            @test solve(allfixed, alg).u == [1.0, 1.0]
            @test SciMLBase.successful_retcode(solve(allfixed, alg))
            for A in ([1.0 1.0], [1.0 1.0; 2.0 2.0], [1.0 0.0; 0.0 1.0; 1.0 1.0])
                b = A * [0.3, 0.3]
                prob = NonlinearLeastSquaresProblem((u, p) -> A * u - b, [0.0, 0.0]; lb = 0.0, ub = 1.0)
                sol = solve(prob, alg; abstol = 1.0e-9)
                @test SciMLBase.successful_retcode(sol)
                @test norm(sol.resid) < 1.0e-6
                @test all(0 .<= sol.u .<= 1)
            end
            for u0 in ([0.0], SVector(0.0))
                prob = NonlinearProblem((u, p) -> u .^ 2 .- p, u0, 0.25; lb = 0.0, ub = 1.0)
                # A zero derivative at zero is stationary, so start the root search away from it.
                cache = init(remake(prob; u0 = u0 .+ 0.1), alg; abstol = 1.0e-10)
                sol = solve!(cache)
                @test SciMLBase.successful_retcode(sol)
                @test sol.u isa typeof(u0)
                @test sol.u ≈ u0 .+ 0.5 atol = 1.0e-7
                reinit!(cache, u0 .+ 0.2; p = 0.64)
                again = solve!(cache)
                @test SciMLBase.successful_retcode(again)
                @test again.u ≈ u0 .+ 0.8 atol = 1.0e-7
                @test_throws ArgumentError reinit!(cache, u0 .+ 2)
            end
            f! = (r, u, p) -> (r .= [u[1] - 2, u[2] - 0.5, 1.0])
            prob = NonlinearLeastSquaresProblem(
                NonlinearFunction(f!; resid_prototype = zeros(3)), [0.0, 0.0]; lb = 0.0, ub = 1.0
            )
            sol = solve(prob, alg)
            @test SciMLBase.successful_retcode(sol)
            @test sol.u ≈ [1.0, 0.5] atol = 1.0e-6
            f = NonlinearFunction((u, p) -> u .- 2; jac = (u, p) -> sparse(I, 2, 2))
            sol = solve(NonlinearLeastSquaresProblem(f, zeros(2); lb = 0.0, ub = 1.0), alg)
            @test SciMLBase.successful_retcode(sol)
            @test sol.u ≈ ones(2) atol = 1.0e-6
            calls = Ref(0)
            f = (u, p) -> (calls[] += 1; u)
            @test_throws ArgumentError init(NonlinearProblem(f, [2.0]; lb = 0.0, ub = 1.0), alg)
            @test calls[] == 0
            @test solve(NonlinearProblem((u, p) -> u - 1, 0.5), alg; maxiters = 0).retcode == ReturnCode.MaxIters
        end
    end
end

@testset "Reflective strict feasibility and nonlinear model" begin
    f = function (u, p)
        @test -2 < u[1] < 0.8
        @test -1 < u[2] < 2
        return [10 * (u[2] - u[1]^2), 1 - u[1]]
    end
    for ad in (AutoForwardDiff(), AutoFiniteDiff())
        # Finite differences may evaluate on a bound; the residual domain includes it.
        residual = ad isa AutoFiniteDiff ? (u, p) -> begin
                @test all([-2.0, -1.0] .<= ForwardDiff.value.(u) .<= [0.8, 2.0])
                [10 * (u[2] - u[1]^2), 1 - u[1]]
            end : f
        prob = NonlinearLeastSquaresProblem(residual, [-2.0, 2.0]; lb = [-2.0, -1.0], ub = [0.8, 2.0])
        sol = solve(prob, TrustRegionReflective(; autodiff = ad); abstol = 1.0e-10)
        @test SciMLBase.successful_retcode(sol)
        @test sol.u ≈ [0.8, 0.64] atol = 2.0e-6
    end
    @test_throws ArgumentError TrustRegionReflective(; initial_trust_radius = 0)
    @test_throws ArgumentError TrustRegionReflective(; gtol = -1)
end

@testset "Reflective state types and sparse differencing" begin
    for u0 in (Float32[0.1, 0.1], reshape([0.1, 0.1], 1, 2))
        target = oftype(u0, u0 .* 0 .+ 0.5)
        prob = NonlinearProblem((u, p) -> u .- p, u0, target; lb = 0, ub = 1)
        sol = solve(prob, TrustRegionReflective(); abstol = 1.0e-6)
        @test SciMLBase.successful_retcode(sol)
        @test typeof(sol.u) == typeof(u0)
        @test sol.u ≈ target atol = 1.0e-6
    end
    f = (u, p) -> begin
        @test all(0 .<= u .<= 1)
        u .- 2
    end
    prob = NonlinearLeastSquaresProblem(f, [0.0, 1.0]; lb = 0.0, ub = 1.0)
    sol = solve(prob, TrustRegionReflective(; autodiff = AutoSparse(AutoFiniteDiff())))
    @test SciMLBase.successful_retcode(sol)
    @test sol.u ≈ ones(2) atol = 1.0e-6
end

@testset "Constrained LM model" begin
    A = [1.0 2.0; 3.0 1.0; 0.0 1.0]
    b = [2.0, -1.0, 0.5]
    for damping in (1.0e-6, 0.01, 1.0), lo in ([-0.1, -0.2], [0.0, 0.0])
        hi = [0.3, 0.4]
        prob = NonlinearLeastSquaresProblem((u, p) -> A * u - b, zeros(2); lb = lo, ub = hi)
        cache = init(prob, BoundedLevenbergMarquardt())
        s = NonlinearSolveFirstOrder._box_constrained_lsq(cache, A, -b, lo, hi, damping)
        g = A' * (A * s - b) + damping * s
        @test all(lo .<= s .<= hi)
        @test norm(s - clamp.(s - g, lo, hi), Inf) < 1.0e-7
    end
    f = (u, p) -> [10 * (u[2] - u[1]^2), 1 - u[1]]
    prob = NonlinearLeastSquaresProblem(f, [-1.2, 1.0]; lb = [-2.0, -1.0], ub = [0.8, 2.0])
    sol = solve(prob, BoundedLevenbergMarquardt(); abstol = 1.0e-10)
    @test SciMLBase.successful_retcode(sol)
    @test sol.u ≈ [0.8, 0.64] atol = 2.0e-6
    cache = init(prob, BoundedLevenbergMarquardt(; damping = 0.02))
    solve!(cache)
    reinit!(cache, prob.u0)
    @test cache.damping == 0.02
    @test SciMLBase.successful_retcode(solve!(cache))
    @test_throws ArgumentError BoundedLevenbergMarquardt(; damping = 0)
    @test_throws ArgumentError BoundedLevenbergMarquardt(; max_backtracks = 0)
end

@testset "Rectangular dogleg and rank deficiency" begin
    prob = NonlinearProblem((u, p) -> u .- 4, [0.0, 0.0]; lb = -10.0, ub = 10.0)
    cache = init(prob, Dogbox(; initial_trust_radius = 1))
    step!(cache)
    @test cache.u ≈ [1.0, 1.0]
    @test norm(cache.u) > 1
    f = (u, p) -> [u[1]^2, u[1]^2]
    singular = NonlinearLeastSquaresProblem(f, [1.0, 0.0]; lb = [0.0, -1.0], ub = [2.0, 1.0])
    limited = solve(singular, Dogbox(; gtol = 0); abstol = 1.0e-12, maxiters = 5)
    @test limited.retcode == ReturnCode.MaxIters
    @test limited.u[1] ≈ 1 / 32
    converged = solve(singular, Dogbox(; gtol = 0); abstol = 1.0e-12)
    @test SciMLBase.successful_retcode(converged)
    @test converged.stats.nsteps > 10
    @test norm(converged.resid) <= 1.0e-12
    f = (u, p) -> [10 * (u[2] - u[1]^2), 1 - u[1]]
    sol = solve(NonlinearLeastSquaresProblem(f, [-1.2, 1.0]; lb = [-2.0, -1.0], ub = [0.8, 2.0]), Dogbox())
    @test SciMLBase.successful_retcode(sol)
    @test sol.u ≈ [0.8, 0.64] atol = 2.0e-6
end

@testset "Reduced Gauss–Newton globalizations" begin
    A = [1.0 10.0; 0.0 1.0]
    b = [0.0, 1.0]
    prob = NonlinearLeastSquaresProblem((u, p) -> A * u - b, [0.1, 0.1]; lb = 0.0, ub = 1.0)
    for globalization in (:linesearch, :trustregion, :trustregion_linesearch)
        sol = solve(prob, BoundedGaussNewton(; globalization))
        @test SciMLBase.successful_retcode(sol)
        @test sol.u ≈ [0.0, 1 / 101] atol = 1.0e-6
        @test norm(sol.u - clamp.(sol.u - A' * sol.resid, 0, 1), Inf) <= 1.0e-7
    end
    prob = NonlinearProblem((u, p) -> exp(u) - 2, -2.0; lb = -10.0, ub = 10.0)
    plain = init(prob, BoundedGaussNewton(; globalization = :trustregion, initial_trust_radius = 1000))
    fallback = init(prob, BoundedGaussNewton(; globalization = :trustregion_linesearch, initial_trust_radius = 1000))
    step!(plain)
    step!(fallback)
    @test plain.u == prob.u0
    @test fallback.u > prob.u0
    @test abs(fallback.fu) < abs(plain.fu)
    @test SciMLBase.successful_retcode(solve!(fallback))
    @test_throws ArgumentError BoundedGaussNewton(; globalization = :invalid)
    @test_throws ArgumentError BoundedGaussNewton(; max_backtracks = 0)
end

@testset "Scalar and vector residual combinations" begin
    for constructor in native_bounded_algorithms, ad in (AutoForwardDiff(), AutoFiniteDiff())
        @testset "$(constructor), $(ad)" begin
            prob = NonlinearLeastSquaresProblem(
                (u, p) -> [u - p, 1.0], 0.0, 2.0; lb = -1.0, ub = 1.0
            )
            cache = init(prob, constructor(; autodiff = ad))
            sol = solve!(cache)
            @test SciMLBase.successful_retcode(sol)
            @test sol.u isa Float64
            @test sol.u ≈ 1.0 atol = 1.0e-6
            reinit!(cache, 0.0; p = -0.5)
            @test solve!(cache).u ≈ -0.5 atol = 1.0e-6
            prob = NonlinearLeastSquaresProblem(
                (u, p) -> u[1] + u[2] - 0.6, [0.0, 0.0]; lb = 0.0, ub = 1.0
            )
            sol = solve(prob, constructor(; autodiff = ad))
            @test SciMLBase.successful_retcode(sol)
            @test sol.resid isa Float64
            @test abs(sol.resid) < 1.0e-7
        end
    end
end

@testset "Analytic Jacobians and in-place residuals" begin
    for constructor in native_bounded_algorithms
        for analytic in (false, true)
            f! = (r, u, p) -> (r .= (u[1] - p, 1.0))
            jac! = analytic ? ((J, u, p) -> (J .= reshape([1.0, 0.0], 2, 1))) : nothing
            f = NonlinearFunction{true}(f!; jac = jac!, resid_prototype = zeros(2))
            prob = NonlinearLeastSquaresProblem(f, [0.0], 2.0; lb = -1.0, ub = 1.0)
            cache = init(prob, constructor())
            sol = solve!(cache)
            @test SciMLBase.successful_retcode(sol)
            @test sol.u ≈ [1.0] atol = 1.0e-6
            reinit!(cache, [0.0]; p = -0.5)
            @test solve!(cache).u ≈ [-0.5] atol = 1.0e-6
        end
        for (f, jac, u0) in (
                ((u, p) -> [u - 2, 1.0], (u, p) -> [1.0, 0.0], 0.0),
                ((u, p) -> u[1] + u[2] - 3, (u, p) -> [1.0 1.0], [0.0, 0.0]),
            )
            prob = NonlinearLeastSquaresProblem(NonlinearFunction(f; jac), u0; lb = 0.0, ub = 1.0)
            sol = solve(prob, constructor())
            @test SciMLBase.successful_retcode(sol)
            @test sol.u ≈ one.(u0) atol = 1.0e-6
        end
    end
end

@testset "Native bounds preserve sparse and matrix-free linear solves" begin
    for constructor in native_bounded_algorithms
        @testset "$(constructor), $(representation)" for representation in (:sparse, :operator, :normal, :sparse_iterative)
            sparse_representation = representation in (:sparse, :sparse_iterative)
            n = 32
            target = collect(range(-0.5, 1.5; length = n))
            products = Ref(0)
            f! = (r, u, p) -> (r .= u .- p)
            jac! = sparse_representation ? ((J, u, p) -> (J[diagind(J)] .= 1)) :
                ((J, u, p) -> error("The Krylov path must not materialize the Jacobian"))
            product! = (w, v, u, p) -> begin
                products[] += 1
                w .= v
            end
            f = NonlinearFunction(
                f!; jac = jac!, jvp = product!, vjp = product!,
                jac_prototype = spdiagm(0 => ones(n))
            )
            prob = NonlinearLeastSquaresProblem(f, fill(0.5, n), target; lb = 0.0, ub = 1.0)
            precs_calls = Ref(0)
            precs = (A, p) -> begin
                precs_calls[] += 1
                if precs_calls[] == 1
                    @test p.u == fill(0.5, n)
                    @test p.p == target
                end
                @test A isa SparseMatrixCSC
                return Diagonal(ones(n)), LinearAlgebra.I
            end
            linsolve = representation === :sparse ? nothing :
                (
                    representation === :sparse_iterative ? LinearSolve.KrylovJL_GMRES(; precs) :
                    (representation === :normal ? LinearSolve.KrylovJL_GMRES() : LinearSolve.KrylovJL_LSMR())
                )
            cache = init(
                prob, constructor(; linsolve, concrete_jac = representation === :sparse_iterative);
                linsolve_kwargs = (; abstol = 0.0, reltol = 1.0e-10)
            )
            J = NonlinearSolveBase.reused_jacobian(cache.jac_cache, cache.u)
            linear_cache = NonlinearSolveBase.get_linear_cache(cache)
            @test sparse_representation ? J isa SparseMatrixCSC : J isa SciMLOperators.AbstractSciMLOperator
            @test sparse_representation ? linear_cache.A isa SparseMatrixCSC : linear_cache.A isa SciMLOperators.AbstractSciMLOperator
            sol = solve!(cache)
            @test SciMLBase.successful_retcode(sol)
            @test sol.u ≈ clamp.(target, 0.0, 1.0) atol = 1.0e-6
            @test sol.stats.nsolve > 0
            @test sparse_representation || products[] > 0
            @test representation !== :sparse_iterative || precs_calls[] > 0
            reinit!(cache, fill(0.5, n); p = reverse(target))
            @test solve!(cache).u ≈ clamp.(reverse(target), 0.0, 1.0) atol = 1.0e-6
            @test NonlinearSolveBase.get_linear_cache(cache) === linear_cache
        end
    end
end

@testset "Sparse AD and feasible operator differences" begin
    for constructor in native_bounded_algorithms, ad in (AutoForwardDiff(), AutoFiniteDiff()), concrete in (false, true)
        @testset "$(constructor), $(ad), concrete=$(concrete)" begin
            lb, ub = [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 1.0]
            f = (u, p) -> begin
                @test all(lb .<= ForwardDiff.value.(u) .<= ub)
                exp.(u) .- p
            end
            target = [1.0, 1.3, 4.0, 1.7]
            prob = NonlinearLeastSquaresProblem(
                NonlinearFunction(f; jac_prototype = spdiagm(0 => ones(4))),
                zeros(4), target; lb, ub
            )
            # The residual's test assertions are not differentiable by reverse AD.
            vjp_autodiff = ad isa AutoFiniteDiff ? nothing : AutoFiniteDiff()
            cache = init(prob, constructor(; autodiff = ad, vjp_autodiff, linsolve = LinearSolve.KrylovJL_LSMR(), concrete_jac = concrete))
            if ad isa AutoFiniteDiff
                @test cache.alg.jvp_autodiff isa AutoFiniteDiff
                @test cache.alg.vjp_autodiff isa AutoFiniteDiff
            end
            @test concrete ? NonlinearSolveBase.reused_jacobian(cache.jac_cache, cache.u) isa SparseMatrixCSC :
                NonlinearSolveBase.reused_jacobian(cache.jac_cache, cache.u) isa SciMLOperators.AbstractSciMLOperator
            sol = solve!(cache)
            @test SciMLBase.successful_retcode(sol)
            @test sol.u ≈ clamp.(log.(target), lb, ub) atol = 1.0e-6
        end
    end
end

@testset "Matrix-free mixed state and residual shapes" begin
    for constructor in native_bounded_algorithms, prob in (
                NonlinearLeastSquaresProblem((u, p) -> [u - p, 1.0], 0.0, 2.0; lb = 0.0, ub = 1.0),
                NonlinearLeastSquaresProblem((u, p) -> sum(u) - p, [0.0, 0.0], 3.0; lb = 0.0, ub = 1.0),
            )
        cache = init(prob, constructor(; linsolve = LinearSolve.KrylovJL_LSMR()))
        @test NonlinearSolveBase.reused_jacobian(cache.jac_cache, cache.u) isa SciMLOperators.AbstractSciMLOperator
        sol = solve!(cache)
        @test SciMLBase.successful_retcode(sol)
        @test sol.u ≈ one.(prob.u0) atol = 1.0e-6
    end
end

@testset "User-supplied Jacobian operators" begin
    for constructor in native_bounded_algorithms, analytic in (false, true)
        product = (v, u, p, t) -> p .* v
        prototype = analytic ? SciMLOperators.FunctionOperator(
                product, zeros(2); op_adjoint = product, u = zeros(2), p = [2.0, 3.0], islinear = true
            ) : nothing
        prob = NonlinearLeastSquaresProblem(
            NonlinearFunction(
                (u, p) -> p .* u .- 1; jac_prototype = prototype,
                jac = analytic ? (
                        (u, p) -> begin
                            SciMLOperators.update_coefficients!(prototype, u, p, 0.0)
                            prototype
                        end
                    ) : nothing
            ),
            zeros(2), [2.0, 3.0]; lb = 0.0, ub = 1.0
        )
        cache = init(prob, constructor(; linsolve = LinearSolve.KrylovJL_LSMR()))
        @test analytic ? NonlinearSolveBase.reused_jacobian(cache.jac_cache, cache.u) === prototype :
            NonlinearSolveBase.reused_jacobian(cache.jac_cache, cache.u) isa SciMLOperators.AbstractSciMLOperator
        sol = solve!(cache)
        @test SciMLBase.successful_retcode(sol)
        @test sol.u ≈ [0.5, 1 / 3] atol = 1.0e-6
        reinit!(cache, zeros(2); p = [4.0, 5.0])
        @test solve!(cache).u ≈ [0.25, 0.2] atol = 1.0e-6
    end
end

@testset "Convertible Jacobian operators preserve linear representations" begin
    for constructor in native_bounded_algorithms, (linsolve, concrete) in (
                (nothing, false), (LinearSolve.LUFactorization(), false),
                (LinearSolve.KrylovJL_LSMR(), false), (LinearSolve.KrylovJL_LSMR(), true),
            )
        A = spdiagm(0 => [2.0, 3.0])
        prototype = SciMLOperators.MatrixOperator(A)
        prob = NonlinearLeastSquaresProblem(
            NonlinearFunction((u, p) -> A * u - p; jac = (u, p) -> prototype, jac_prototype = prototype),
            zeros(2), ones(2); lb = 0.0, ub = 1.0
        )
        cache = init(prob, constructor(; linsolve, concrete_jac = concrete))
        @test NonlinearSolveBase.reused_jacobian(cache.jac_cache, cache.u) === prototype
        matrix = concrete || linsolve === nothing || linsolve isa LinearSolve.LUFactorization
        linear = NonlinearSolveBase.get_linear_cache(cache)
        @test matrix ? linear.A isa SparseMatrixCSC : linear.A isa SciMLOperators.AbstractSciMLOperator
        sol = solve!(cache)
        @test SciMLBase.successful_retcode(sol)
        @test sol.u ≈ [0.5, 1 / 3] atol = 1.0e-6
        reinit!(cache, zeros(2); p = [0.5, 0.75])
        @test solve!(cache).u ≈ [0.25, 0.25] atol = 1.0e-6
        @test NonlinearSolveBase.get_linear_cache(cache) === linear
    end
end

@testset "Shared bounded globalization policies" begin
    linear = NonlinearProblem((u, p) -> u .- 2, [0.0]; lb = -10.0, ub = 10.0)
    cache = init(
        linear, BoundedTrustRegion(;
            initial_trust_radius = 0.25, max_trust_radius = 0.6,
            expand_threshold = 0.8, expand_factor = 3
        )
    )
    step!(cache)
    @test cache.u ≈ [0.25]
    @test cache.trustregion_cache.last_step_accepted
    @test cache.trustregion_cache.trust_region == 0.6
    reinit!(cache, [0.0])
    @test cache.trustregion_cache.trust_region == 0.25

    prob = NonlinearProblem((u, p) -> atan.(10 .* u), [1.0]; lb = -100.0, ub = 100.0)
    newton_length = atan(10.0) / (10 / 101)
    algorithms = (
        BoundedTrustRegion(; initial_trust_radius = 20, shrink_factor = 1 // 8),
        Dogbox(; initial_trust_radius = 20),
        BoundedGaussNewton(; globalization = :trustregion, initial_trust_radius = 20),
    )
    for alg in algorithms
        cache = init(prob, alg)
        step!(cache)
        @test cache.u == prob.u0
        @test cache.fu == atan.(10 .* prob.u0)
        if alg isa typeof(first(algorithms))
            @test !cache.trustregion_cache.last_step_accepted
            @test cache.trustregion_cache.shrink_counter == 1
            @test cache.trustregion_cache.trust_region == 2.5
        else
            @test cache.radius ≈ newton_length / 4
        end
    end
end

@testset "Shared projected line-search cache" begin
    for alg in (
            BoundedLevenbergMarquardt(), BoundedGaussNewton(),
            BoundedGaussNewton(; globalization = :trustregion_linesearch),
        )
        evaluations = Ref(0)
        f(u, p) = (evaluations[] += 1; @assert all(0 .<= u .<= 2); u .- p)
        nf = NonlinearFunction{false}(f; jac = (u, p) -> Matrix{Float64}(I, 2, 2))
        prob = NonlinearLeastSquaresProblem(nf, [0.5, 0.5], [1.0, 1.5]; lb = 0.0, ub = 2.0)
        cache = init(prob, alg; abstol = 1.0e-10)
        linesearch_cache = cache.linesearch_cache
        sol = solve!(cache)
        @test SciMLBase.successful_retcode(sol.retcode)
        @test sol.u ≈ [1.0, 1.5] atol = 1.0e-8
        @test sol.stats.nf == evaluations[]
        reinit!(cache; u0 = [0.25, 0.25], p = [1.5, 1.0])
        sol = solve!(cache)
        @test SciMLBase.successful_retcode(sol.retcode)
        @test sol.u ≈ [1.5, 1.0] atol = 1.0e-8
        @test cache.linesearch_cache === linesearch_cache
    end
end


@testset "Geometric Armijo in the first-order solve loop" begin
    prob = NonlinearProblem((u, p) -> u .^ 2 .- 2, [0.1, 0.1])
    sol = solve(prob, NewtonRaphson(; linesearch = ArmijoLineSearch()); abstol = 1.0e-10)
    @test SciMLBase.successful_retcode(sol.retcode)
    @test sol.u ≈ fill(sqrt(2), 2) atol = 1.0e-8
end

@testset "Bounded trust-region finite differences" begin
    for u0 in (1.0, [1.0], SVector(1.0), ones(2, 2), SMatrix{2, 2}(ones(2, 2))), fixed in (false, true)
        f = (u, p) -> begin
            all((fixed ? 1.0 : 0.0) .<= u .<= 1.0) || throw(DomainError(u))
            u .- p
        end
        prob = NonlinearLeastSquaresProblem(f, u0, 2.0; lb = fixed ? 1.0 : 0.0, ub = 1.0)
        cache = init(prob, BoundedTrustRegion(; autodiff = AutoFiniteDiff()))
        @test (NonlinearSolveBase.reused_jacobian(cache.jac_cache, cache.u) isa Number) == (u0 isa Number)
        sol = solve!(cache)
        @test SciMLBase.successful_retcode(sol)
        @test sol.u ≈ u0
        @test sol.u isa typeof(u0)
        reinit!(cache, u0; p = 0.25)
        again = solve!(cache)
        @test SciMLBase.successful_retcode(again)
        @test again.u ≈ (fixed ? u0 : 0.25 .* u0)
    end
    for inplace in (false, true), representation in (:dense, :sparse, :operator),
            mixed in (false, true)
        lb, ub = [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]
        f = (u, p) -> begin
            all(lb .<= ForwardDiff.value.(u) .<= ub) || throw(DomainError(u))
            exp.(u) .- p
        end
        fun = inplace ? ((r, u, p) -> (r .= f(u, p))) : f
        prototype = representation === :sparse ? spdiagm(0 => ones(3)) : nothing
        prob = NonlinearLeastSquaresProblem(
            NonlinearFunction{inplace}(fun; jac_prototype = prototype),
            ones(3) .* ub, [1.0, 1.3, 4.0]; lb, ub
        )
        alg = BoundedTrustRegion(;
            autodiff = mixed ? AutoForwardDiff() : AutoFiniteDiff(),
            vjp_autodiff = AutoFiniteDiff(),
            jvp_autodiff = AutoFiniteDiff(),
            linsolve = representation === :operator ? LinearSolve.KrylovJL_LSMR() : nothing
        )
        cache = init(prob, alg)
        jac_cache = cache.jac_cache
        J = NonlinearSolveBase.reused_jacobian(jac_cache, cache.u)
        @test representation === :operator ? J isa SciMLOperators.AbstractSciMLOperator :
            representation === :sparse ? J isa SparseMatrixCSC : J isa Matrix
        sol = solve!(cache)
        @test SciMLBase.successful_retcode(sol)
        @test sol.u ≈ clamp.(log.(prob.p), lb, ub) atol = 1.0e-6
        reinit!(cache, ub; p = [1.0, 4.0, 1.7])
        again = solve!(cache)
        @test SciMLBase.successful_retcode(again)
        @test again.u ≈ clamp.(log.([1.0, 4.0, 1.7]), lb, ub) atol = 1.0e-6
        @test cache.jac_cache === jac_cache
        @test NonlinearSolveBase.reused_jacobian(cache.jac_cache, cache.u) isa typeof(J)
    end
end
