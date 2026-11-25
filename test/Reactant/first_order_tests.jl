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
        u = ones(Float32, 2)
        results = (step_newton(u, Val(limit)), Reactant.@jit(step_newton(Reactant.to_rarray(u), Val(limit))))
        for (x, fx, nsteps) in results
            @test Array(x) == fill(iterate, 2)
            @test Array(fx) == fill(residual, 2)
            @test Int(nsteps) == limit
        end
    end
end

@testset "Failed initialization preserves the cache state" begin
    f(u, p) = u .- 1
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
        u = Float32[2, 2]
        results = (failed_init(u, alg), Reactant.@jit(failed_init(Reactant.to_rarray(u), alg)))
        for (x, fx, retcode, nsteps, evaluations) in results
            @test Array(x) == Float32[2, 2]
            @test Array(fx) == Float32[1, 1]
            @test retcode == ReturnCode.InitialFailure
            @test (isnothing(nsteps) ? nothing : Int(nsteps)) === expected_steps
            @test Array(evaluations) == Float32[0, 0]
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

@testset "Augmented trust-region right-hand side" begin
    function augmented_rhs(Dp, λ)
        return NonlinearSolveBase._more_augmented_qrhs!(similar(Dp, 4), Dp, λ, 2)
    end
    rhs = Reactant.@jit augmented_rhs(
        Reactant.to_rarray(Float32[2, 4]), Reactant.ConcreteRNumber(4.0f0)
    )
    @test Array(rhs) == Float32[0, 0, 1, 2]
end

struct ReturnIterate end

function NonlinearSolveBase.InternalAPI.step!(cache, ::ReturnIterate)
    NonlinearSolveBase.InternalAPI.step!(cache)
    return copy(cache.u)
end

@testset "Public step! with an iterate payload" begin
    f(u, p) = u .^ 2 .- 2
    jac(u, p) = Diagonal(2 .* u)
    function returning_step(u)
        cache = init(NonlinearProblem(NonlinearFunction(f; jac), u), NewtonRaphson())
        return step!(cache, ReturnIterate())
    end
    u = Float32[1]
    @test returning_step(u) == Float32[1.5]
    err = ArgumentError("compiled step! requires a nothing return value when termination is data-dependent")
    @test_throws err Reactant.@compile returning_step(Reactant.to_rarray(u))
end

struct AuxiliaryTarget{A}
    value::A
end

NonlinearSolveBase.initialization_alg(::NonlinearProblem{U, I, <:AuxiliaryTarget}, ad) where {U, I} = NewtonRaphson()

struct UncachedNewton <: NonlinearSolveBase.AbstractNonlinearSolveAlgorithm end

function SciMLBase.__solve(prob::NonlinearProblem, ::UncachedNewton, args...; kwargs...)
    return SciMLBase.__solve(prob, NewtonRaphson(), args...; kwargs...)
end

@testset "Auxiliary initialization selects the outer solve" begin
    function initialized_problem(u, target)
        residual(x, p) = x .^ 2 .- p.value
        jac(x, p) = diagm(2 .* x)
        aux = NonlinearProblem(NonlinearFunction(residual; jac), one.(u), AuxiliaryTarget(target); maxiters = 1)
        data = SciMLBase.OverrideInitData(aux, nothing, sol -> sol.u, nothing)
        outer = NonlinearFunction((x, p) -> x .- 2; jac = (x, p) -> Diagonal(one.(x)), initialization_data = data)
        return NonlinearProblem(outer, u)
    end
    function initialized_solve(u, target, alg)
        cache = init(initialized_problem(u, target), alg)
        mapped = copy(NonlinearSolveBase.get_u(cache))
        sol = solve!(cache)
        return mapped, sol.u, sol.resid, sol.retcode, hasproperty(cache, :nsteps) ? cache.nsteps : nothing
    end
    poly = NonlinearSolvePolyAlgorithm((NewtonRaphson(), NewtonRaphson()))
    for alg in (NewtonRaphson(), poly, UncachedNewton())
        u = fill(3.0f0, 2)
        for target in (-1.0f0, 1.0f0)
            p = fill(target, 2)
            for result in (initialized_solve(u, p, alg), Reactant.@jit(initialized_solve(Reactant.to_rarray(u), Reactant.to_rarray(p), alg)))
                mapped, x, fx, retcode, nsteps = result
                @test Array(mapped) == fill(target < 0 ? 0.0f0 : 1.0f0, 2)
                @test Array(x) == fill(target < 0 ? 0.0f0 : 2.0f0, 2)
                @test Array(fx) == Array(x) .- 2
                @test retcode == (target < 0 ? ReturnCode.InitialFailure : ReturnCode.Success)
                if target < 0 && !isnothing(nsteps)
                    @test Int(nsteps) == 0
                end
            end
        end
    end

    direct(u, target) = solve(initialized_problem(u, target), poly)
    u = fill(3.0f0, 2)
    run = Reactant.@compile direct(Reactant.to_rarray(u), Reactant.to_rarray(u))
    for target in (-1.0f0, 1.0f0)
        sol = run(Reactant.to_rarray(u), Reactant.to_rarray(fill(target, 2)))
        @test Array(sol.u) == fill(target < 0 ? 0.0f0 : 2.0f0, 2)
        @test Array(sol.resid) == Array(sol.u) .- 2
        @test sol.retcode == (target < 0 ? ReturnCode.InitialFailure : ReturnCode.Success)
    end
end
