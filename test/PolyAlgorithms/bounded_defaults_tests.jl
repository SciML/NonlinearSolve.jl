using NonlinearSolve, NonlinearSolveBase, SciMLBase, StaticArrays, SparseArrays
using LinearSolve, SciMLOperators, LinearAlgebra, ForwardDiff

@testset "Bounded defaults preserve physical coordinates" begin
    for constructor in (NonlinearProblem, NonlinearLeastSquaresProblem),
            u0 in (0.0, [0.0, 1.0], @SVector([0.0, 1.0]), zeros(2, 2), Float32[0, 1])
        target = zero(u0) .+ eltype(u0)(0.75)
        prob = constructor((u, p) -> u .- p, u0, target; lb = 0, ub = 1)
        for cached in (false, true)
            if cached
                cache = init(prob; abstol = 1.0e-7, reltol = 1.0e-7)
                @test cache.alg isa NonlinearSolvePolyAlgorithm
                @test SciMLBase.allowsbounds(cache.alg)
                @test all(c -> c.prob.lb == 0 && c.prob.ub == 1, cache.caches)
                sol = solve!(cache)
            else
                sol = solve(prob; abstol = 1.0e-7, reltol = 1.0e-7)
            end
            @test SciMLBase.successful_retcode(sol)
            @test sol.u ≈ target atol = 1.0e-6
            @test typeof(sol.u) == typeof(u0)
        end
    end
end

@testset "Boundary stationarity and root failure" begin
    for bounds in ((; lb = 0.0), (; ub = 1.0), (; lb = 0.0, ub = 1.0), (; lb = 1.0, ub = 1.0))
        target = haskey(bounds, :ub) ? 2.0 : -1.0
        optimum = haskey(bounds, :ub) ? 1.0 : 0.0
        for constructor in (NonlinearProblem, NonlinearLeastSquaresProblem)
            prob = constructor((u, p) -> u .- p, [optimum], target; bounds...)
            for cached in (false, true)
                sol = cached ? solve!(init(prob; maxiters = 30)) : solve(prob; maxiters = 30)
                @test SciMLBase.successful_retcode(sol) == (constructor === NonlinearLeastSquaresProblem)
                @test sol.u ≈ [optimum]
                @test sol.resid ≈ [optimum - target]
            end
        end
    end
end

@testset "Bounded polyalgorithm traits and explicit transformations" begin
    bounded = FastShortcutBoundedPolyalg()
    mixed = NonlinearSolvePolyAlgorithm((NewtonRaphson(), BoundedGaussNewton()))
    @test SciMLBase.allowsbounds(bounded)
    @test !SciMLBase.allowsbounds(mixed)
    prob = NonlinearProblem((u, p) -> u .- p, [0.25], 0.75; lb = 0.0, ub = 1.0)
    for alg in (NewtonRaphson(), mixed)
        cache = init(prob, alg)
        @test cache.prob.lb === nothing
        @test cache.prob.ub === nothing
        @test solve!(cache).u ≈ [0.75]
    end
    unbounded = SciMLBase.remake(prob; lb = nothing, ub = nothing)
    @test !SciMLBase.allowsbounds(init(unbounded).alg)
    @test NonlinearSolveBase.initialization_alg(prob, AutoForwardDiff()).algs ==
        FastShortcutBoundedPolyalg(autodiff = AutoForwardDiff()).algs
end

@testset "Bounded caches reinitialize parameters and retain stages" begin
    prob = NonlinearLeastSquaresProblem((u, p) -> u .- p, [0.5, 0.5], [2.0, -1.0]; lb = 0.0, ub = 1.0)
    cache = init(prob)
    @test solve!(cache).u ≈ [1.0, 0.0]
    caches = cache.caches
    reinit!(cache; u0 = [0.5, 0.5], p = [-1.0, 2.0], retain_best = true)
    @test cache.caches === caches
    @test solve!(cache).u ≈ [0.0, 1.0]
    reinit!(cache; u0 = [0.5, 0.5], p = [0.3, 0.7])
    for _ in 1:100
        step!(cache)
        cache.force_stop && break
    end
    @test SciMLBase.successful_retcode(cache.retcode)
    @test NonlinearSolveBase.get_u(cache) ≈ [0.3, 0.7]
end

@testset "Sparse and operator Jacobians" begin
    n = 20
    A = spdiagm(-1 => fill(-0.1, n - 1), 0 => fill(2.0, n), 1 => fill(-0.1, n - 1))
    target = fill(0.75, n)
    f!(r, u, p) = mul!(r, A, u - p)
    jac!(J, u, p) = copyto!(J, A)
    jvp!(w, v, u, p) = mul!(w, A, v)
    vjp!(w, v, u, p) = mul!(w, transpose(A), v)
    for matrix_free in (false, true)
        nf = NonlinearFunction(f!; jac = jac!, jac_prototype = copy(A), jvp = jvp!, vjp = vjp!)
        prob = NonlinearLeastSquaresProblem(nf, fill(0.1, n), target; lb = 0.0, ub = 1.0)
        alg = FastShortcutBoundedPolyalg(linsolve = matrix_free ? KrylovJL_LSMR() : nothing)
        cache = init(prob, alg; abstol = 1.0e-9)
        for child in cache.caches
            J = NonlinearSolveBase.reused_jacobian(child.jac_cache, child.u)
            @test matrix_free ? J isa SciMLOperators.AbstractSciMLOperator : J isa SparseMatrixCSC
        end
        sol = solve!(cache)
        @test SciMLBase.successful_retcode(sol)
        @test sol.u ≈ target atol = 1.0e-7
    end
end

@testset "Default initial guesses are projected without mutation" begin
    for constructor in (NonlinearProblem, NonlinearLeastSquaresProblem), cached in (false, true),
            alg in (nothing, FastShortcutBoundedPolyalg())
        u0 = [-2.0, 3.0]
        prob = constructor((u, p) -> u .- p, u0, [0.25, 0.75]; lb = 0.0, ub = 1.0)
        sol = cached ? solve!(init(prob, alg)) : solve(prob, alg)
        @test SciMLBase.successful_retcode(sol)
        @test sol.u ≈ [0.25, 0.75]
        @test u0 == [-2.0, 3.0]
        @test_throws ArgumentError init(prob, BoundedTrustRegion())
    end
end

@testset "Default bounded active-set sensitivities" begin
    solution(p) = solve(
        NonlinearLeastSquaresProblem(
            (u, p) -> u .- p,
            [0.5, 0.5], p; lb = 0.0, ub = 1.0
        )
    ).u
    @test ForwardDiff.jacobian(solution, [2.0, 0.25]) ≈ [0.0 0.0; 0.0 1.0]
end

@testset "Bounded default fallback and retained winner" begin
    for constructor in (NonlinearProblem, NonlinearLeastSquaresProblem)
        prob = constructor((u, p) -> u .- 5, zeros(2); lb = 0.0, ub = 10.0)
        alg = FastShortcutBoundedPolyalg()
        @test !SciMLBase.successful_retcode(solve(prob, alg.algs[1]; maxiters = 1))
        sol = solve(prob; maxiters = 1)
        @test SciMLBase.successful_retcode(sol)
        @test sol.u ≈ [5.0, 5.0]
        cache = init(prob; maxiters = 1)
        @test SciMLBase.successful_retcode(solve!(cache))
        @test cache.best == 2
        reinit!(cache; u0 = zeros(2), retain_best = true)
        @test cache.current == 2
        @test SciMLBase.successful_retcode(solve!(cache))
    end
end

@testset "Initialize bounded default in physical coordinates" begin
    initprob = NonlinearProblem((u, p) -> u .- 0.75, [0.5]; lb = 0.0, ub = 1.0)
    calls = Ref(0)
    update_init! = (prob, context) -> (calls[] += 1; nothing)
    initdata = SciMLBase.OverrideInitData(initprob, update_init!, sol -> sol.u, (_, sol) -> 0.75)
    residual(u, p) = (p == 0.75 || error("initializer has not run"); u .- p)
    nf = NonlinearFunction(residual; initialization_data = initdata)
    for constructor in (NonlinearProblem, NonlinearLeastSquaresProblem), cached in (false, true)
        prob = constructor(nf, [0.1], 0.0; lb = 0.0, ub = 1.0)
        before = calls[]
        sol = cached ? solve!(init(prob)) : solve(prob)
        @test SciMLBase.successful_retcode(sol)
        @test sol.u ≈ [0.75]
        @test calls[] == before + 1
    end
end

@testset "Bounded auxiliary initialization projects updated guesses" begin
    for constructor in (NonlinearProblem, NonlinearLeastSquaresProblem),
            start in (-0.5, 1.5), cached in (false, true)
        initprob = constructor((u, p) -> u .- 0.75, [0.5]; lb = 0.0, ub = 1.0)
        update_init! = (prob, context) -> (prob.u0 .= start; nothing)
        initdata = SciMLBase.OverrideInitData(initprob, update_init!, sol -> sol.u, (_, sol) -> 0.75)
        nf = NonlinearFunction((u, p) -> u .- p; initialization_data = initdata)
        prob = constructor(nf, [0.1], 0.0; lb = 0.0, ub = 1.0)
        sol = cached ? solve!(init(prob)) : solve(prob)
        @test SciMLBase.successful_retcode(sol)
        @test sol.u ≈ [0.75]
        @test initprob.u0 == [start]
    end
end

@testset "Bounded default finite differences" begin
    lb, ub = [0.0, 0.0, 0.5], [1.0, 1.0, 0.5]
    f! = function (r, u, p)
        @test all(lb .<= u .<= ub)
        r .= u .- p
    end
    for representation in (:dense, :sparse, :operator)
        prototype = representation === :sparse ? spdiagm(0 => ones(3)) : nothing
        nf = NonlinearFunction(f!; jac_prototype = prototype)
        prob = NonlinearLeastSquaresProblem(nf, [1.0, 0.0, 0.5], [2.0, 0.25, -1.0]; lb, ub)
        alg = FastShortcutBoundedPolyalg(
            autodiff = AutoFiniteDiff(),
            linsolve = representation === :operator ? KrylovJL_LSMR() : nothing
        )
        for cached in (false, true)
            sol = cached ? solve!(init(prob, alg)) : solve(prob, alg)
            @test SciMLBase.successful_retcode(sol)
            @test sol.u ≈ [1.0, 0.25, 0.5]
        end
    end
end

@testset "Default bounded correctors use compatible stages" begin
    H(u, previous, p, cache) = u
    for constructor in (NonlinearProblem, NonlinearLeastSquaresProblem)
        prob = constructor((u, p) -> u .- 5, zeros(2); lb = 0.0, ub = 10.0)
        cache = init(prob; postcondition = H, maxiters = 1)
        @test length(cache.alg.algs) == 1
        @test NonlinearSolveBase.supports_postcondition(cache.alg)
        @test !SciMLBase.successful_retcode(solve!(cache))
        @test !SciMLBase.successful_retcode(solve(prob; postcondition = H, maxiters = 1))
    end
end

@testset "Bounded default respects NoInit" begin
    initprob = NonlinearProblem((u, p) -> u .- 0.75, [0.5]; lb = 0.0, ub = 1.0)
    initdata = SciMLBase.OverrideInitData(initprob, nothing, sol -> sol.u, (_, sol) -> 0.75)
    nf = NonlinearFunction((u, p) -> u .- p; initialization_data = initdata)
    for constructor in (NonlinearProblem, NonlinearLeastSquaresProblem), cached in (false, true)
        prob = constructor(nf, [0.1], 0.25; lb = 0.0, ub = 1.0)
        sol = cached ? solve!(init(prob; initializealg = SciMLBase.NoInit())) :
            solve(prob; initializealg = SciMLBase.NoInit())
        @test SciMLBase.successful_retcode(sol)
        @test sol.u ≈ [0.25]
    end
end
