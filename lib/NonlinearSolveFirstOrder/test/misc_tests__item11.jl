using NonlinearSolveFirstOrder, LinearSolve, SciMLBase, Test
using NonlinearSolveBase: NonlinearSolveBase, AbsNormTerminationMode, get_fu,
    not_terminated, refresh_residual!, supports_deferred_residual
using LineSearch: BackTracking

f!(du, u, p) = (@. du = u^3 - 1.0; nothing)
prob = NonlinearProblem(f!, fill(3.0, 20), nothing)
forcing = EisenstatWalkerForcing2()
alg = NewtonRaphson(; forcing, linsolve = KrylovJL_GMRES())

function expected_eta(p, ηprev, ratio)
    η = p.γ * ratio^p.α
    if p.safeguard
        ηsg = p.γ * ηprev^p.α
        if ηsg > p.safeguard_threshold && ηsg > η
            η = ηsg
        end
    end
    return clamp(η, 0.0, p.ηₘₐₓ)
end

@testset "forcing residual matches the current iterate after each step" begin
    cache = init(prob, alg; abstol = 1.0e-10, reltol = 1.0e-10)
    fc = cache.forcing_cache
    prev_true_ratio = nothing
    prev_η = fc.η
    prev_fu_norm = fc.internalnorm(get_fu(cache))
    @test fc.rnorm == prev_fu_norm
    @test fc.rnorm_prev == prev_fu_norm

    for k in 1:8
        not_terminated(cache) || break
        ratio_used = fc.rnorm / fc.rnorm_prev
        if k == 1
            @test ratio_used == 1
        else
            @test ratio_used ≈ prev_true_ratio
        end
        step!(cache)
        if k == 1
            @test fc.η == fc.p.η₀
        else
            @test fc.η ≈ expected_eta(fc.p, prev_η, ratio_used)
        end
        prev_η = fc.η
        cur = fc.internalnorm(get_fu(cache))
        @test fc.rnorm == cur
        prev_true_ratio = cur / prev_fu_norm
        prev_fu_norm = cur
    end
    @test cache.nsteps ≥ 2
end

@testset "line-search path also records the post-step residual" begin
    ls_alg = NewtonRaphson(;
        forcing, linsolve = KrylovJL_GMRES(), linesearch = BackTracking()
    )
    cache = init(prob, ls_alg; abstol = 1.0e-10, reltol = 1.0e-10)
    fc = cache.forcing_cache
    for _ in 1:8
        not_terminated(cache) || break
        step!(cache)
        @test fc.rnorm == fc.internalnorm(get_fu(cache))
    end
    @test cache.nsteps ≥ 2
end

@testset "deferred residual defers the forcing-norm update" begin
    cache = init(
        prob, alg;
        termination_condition = AbsNormTerminationMode(Base.Fix1(maximum, abs))
    )
    @test supports_deferred_residual(cache)
    fc = cache.forcing_cache
    rnorm0 = fc.rnorm
    fu0 = copy(get_fu(cache))
    step!(cache; evaluate_residual = false)
    @test cache.fu_deferred
    @test get_fu(cache) == fu0
    @test fc.rnorm == rnorm0
    refresh_residual!(cache)
    @test !cache.fu_deferred
    @test fc.rnorm == fc.internalnorm(get_fu(cache))
    @test fc.rnorm != rnorm0
    @test fc.rnorm_prev == rnorm0
end
