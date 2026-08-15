using MultiLevelNonlinearSolve, Test
include("setup_barmodel.jl")

using LinearAlgebra

# T3 — convergence order. `:always` reassembles `S` at every iterate and must stay quadratic;
# `:chord` freezes it after the first assembly and must degrade to linear.
#
# The order is read off the last triple whose residuals are all above the roundoff floor.
# Fitting the floored tail instead would report ≈1 for every method, because once the
# residual reaches machine precision the ratios stop saying anything about the method.
@testset "T3 convergence order — $(label)" for (label, cscale) in
                                               (
        ("homogeneous", ones(40)), ("heterogeneous", HETEROGENEOUS_CSCALE),
    )
    prob, _ = bar_problem(; cscale)
    cache = init(prob, MultiLevelNewton(); abstol = 1.0e-12)
    e = residual_history(cache)
    @test SciMLBase.successful_retcode(cache.retcode)
    @test tail_order(e) > 1.8

    prob_c, model_c = bar_problem(; cscale)
    cache_c = init(prob_c, MultiLevelNewton(; jacobian_reuse = :chord); abstol = 1.0e-12)
    e_c = residual_history(cache_c)
    @test SciMLBase.successful_retcode(cache_c.retcode)
    @test 0.8 < tail_order(e_c) < 1.3
    # Frozen means frozen: exactly one assembly for the whole solve.
    @test model_c.counters.nassembly == 1
    @test cache_c.stats.njacs == 1
    # And it really did take more iterations than the quadratic run, i.e. the two runs are
    # not accidentally identical.
    @test length(e_c) > length(e)
end

# T4 — the guard on the Schur corrector. Scaling `dq/dε` by 1.01 inside `assemble_S!` leaves
# the residual untouched but makes the tangent inconsistent with it, which costs the
# quadratic rate. The perturbation only moves the tangent by a few tenths of a percent, so
# the resulting linear rate is fast: over the first few iterates the sequence still *looks*
# quadratic, and only the tail triples reveal the true order. That is why this is fitted at
# `abstol = 1e-12` on the tail rather than over the whole history.
@testset "T4 a wrong corrector costs the quadratic rate" begin
    prob, _ = bar_problem()
    exact = tail_order(residual_history(init(prob, MultiLevelNewton(); abstol = 1.0e-12)))
    @test exact > 1.8

    for scale in (1.01, 1.05, 1.2)
        prob_w, _ = bar_problem(; corrector_scale = scale)
        cache = init(prob_w, MultiLevelNewton(); abstol = 1.0e-12)
        e = residual_history(cache)
        @test SciMLBase.successful_retcode(cache.retcode)
        @test tail_order(e) < 1.3
    end
end
