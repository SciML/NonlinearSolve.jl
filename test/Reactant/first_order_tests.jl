@testset "Rejected trust-region trial leaves the iterate and residual intact" begin
    f(u, p) = u .^ 2 .- 1
    jac(u, p) = diagm(2 .* u)
    correct(u, old, p, cache) = isnothing(cache) ? u : one.(u)
    function rejected_trial(u)
        return solve(
            NonlinearProblem(NonlinearFunction(f; jac), u),
            TrustRegion(; initial_trust_radius = 100.0f0);
            postcondition = correct, maxiters = 1,
            termination_condition = AbsNormTerminationMode(Base.Fix1(maximum, abs))
        )
    end
    u = fill(0.1f0, 2)
    for sol in (rejected_trial(u), Reactant.@jit(rejected_trial(Reactant.to_rarray(u))))
        @test sol.retcode == ReturnCode.MaxIters
        @test Array(sol.u) == u
        @test Array(sol.resid) ≈ f(u, nothing)
    end
end

@testset "Compilation reports unsupported Jacobian reuse" begin
    f(u, p) = u .^ 2 .- 2
    jac(u, p) = diagm(2 .* u)
    function modified_newton(u)
        return solve(
            NonlinearProblem(NonlinearFunction(f; jac), u),
            NewtonRaphson(; jacobian_reuse = JacobianReuse(; max_age = 10, max_residual_ratio = 1));
            abstol = 0.01f0, maxiters = 2,
            termination_condition = AbsNormTerminationMode(Base.Fix1(maximum, abs))
        )
    end
    u = ones(Float32, 2)
    sol = modified_newton(u)
    @test sol.retcode == ReturnCode.MaxIters
    @test sol.u == fill(1.375f0, 2)
    @test_throws ArgumentError Reactant.@compile modified_newton(Reactant.to_rarray(u))
end

@testset "Public step! advances an active Newton cache once" begin
    f(u, p) = u .^ 2 .- 2
    jac(u, p) = Diagonal(2 .* u)
    function step_newton(u, ::Val{limit}) where {limit}
        cache = init(NonlinearProblem(NonlinearFunction(f; jac), u), NewtonRaphson(); maxiters = limit)
        step!(cache)
        return cache.u, cache.fu, cache.nsteps
    end
    for (limit, iterate, residual) in ((1, 1.5f0, 0.25f0), (0, 1.0f0, -1.0f0))
        u = Float32[1]
        results = (step_newton(u, Val(limit)), Reactant.@jit(step_newton(Reactant.to_rarray(u), Val(limit))))
        for (x, fx, nsteps) in results
            @test Array(x) == [iterate]
            @test Array(fx) == [residual]
            @test Int(nsteps) == limit
        end
    end
end

@testset "Failed initialization preserves the cache state" begin
    f(u, p) = u .- 2
    function jac(u, evaluations)
        evaluations .+= 1
        return Diagonal(one.(u))
    end
    function failed_init(u, alg)
        evaluations = zero.(u)
        cache = init(NonlinearProblem(NonlinearFunction(f; jac), u, evaluations), alg)
        cache.retcode = ReturnCode.InitialFailure
        evaluations .= 0
        sol = solve!(cache)
        nsteps = hasproperty(cache, :nsteps) ? cache.nsteps : nothing
        return sol.u, sol.resid, sol.retcode, nsteps, evaluations
    end
    @testset "$alg" for (alg, expected_steps) in (
            (NewtonRaphson(), 0), (NonlinearSolvePolyAlgorithm((NewtonRaphson(),)), 0),
            (SimpleNewtonRaphson(), nothing),
        )
        u = Float32[3]
        results = (failed_init(u, alg), Reactant.@jit(failed_init(Reactant.to_rarray(u), alg)))
        for (x, fx, retcode, nsteps, evaluations) in results
            @test Array(x) == Float32[3]
            @test Array(fx) == Float32[1]
            @test retcode == ReturnCode.InitialFailure
            @test (isnothing(nsteps) ? nothing : Int(nsteps)) === expected_steps
            @test Array(evaluations) == Float32[0]
        end
    end
end

@testset "Traced safeguarded damping keeps its element type" begin
    function update_damping(x)
        return NonlinearSolveBase._more_update_λ(x, -x, x, x, x / 2, zero(x), 4 * x)
    end
    for T in (Float32, Float64)
        x = Reactant.ConcreteRNumber(one(T))
        λ, l, u = Reactant.@jit update_damping(x)
        @test typeof(λ) === typeof(x)
        @test typeof(l) === typeof(x)
        @test typeof(u) === typeof(x)
        @test T(λ) == T(0.01)
        @test (T(l), T(u)) == (zero(T), one(T))
    end
end
