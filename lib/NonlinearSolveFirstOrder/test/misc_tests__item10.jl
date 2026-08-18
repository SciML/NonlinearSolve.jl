using NonlinearSolveFirstOrder, NonlinearSolveBase, SciMLBase, LinearAlgebra, Test
using NonlinearSolveBase: AbsNormSafeBestTerminationMode, default_gradient_tolerance

# A data fit whose residual is nonzero at the optimum: `‖F‖ ≤ abstol` is unreachable no
# matter how well the solve converges, so only `JᵀF = 0` can certify it.
const TS = collect(range(0.0, 2.0; length = 12))
const YS = [1.0 + 0.7 * t + 0.35 * sin(4t) for t in TS]

expfit(u, p) = @. u[1] * exp(u[2] * TS) - YS
expfit_jac(u, p) = hcat(exp.(u[2] .* TS), u[1] .* TS .* exp.(u[2] .* TS))

function expfit_prob(u0 = [1.0, 0.5])
    f = NonlinearFunction{false}(expfit; jac = expfit_jac, resid_prototype = zeros(length(TS)))
    return NonlinearLeastSquaresProblem(f, u0, nothing)
end

@testset "nonzero-residual least squares reaches ReturnCode.Success" begin
    prob = expfit_prob()
    for alg in (GaussNewton(), LevenbergMarquardt(), TrustRegion())
        sol = solve(prob, alg; maxiters = 1000)
        J = expfit_jac(sol.u, nothing)
        F = expfit(sol.u, nothing)
        @test norm(F, 2) > 1.0e-3                     # the residual test cannot be met
        @test norm(J'F, Inf) < 1.0e-6                 # but the solve is stationary
        @test sol.retcode == ReturnCode.Success
    end
end

@testset "gtol = nothing reproduces the pre-change exit" begin
    prob = expfit_prob()
    off = AbsNormSafeBestTerminationMode(Base.Fix2(norm, 2); max_stalled_steps = 32)
    sol = solve(prob, GaussNewton(); maxiters = 1000, termination_condition = off)
    @test sol.retcode != ReturnCode.Success
end

@testset "the criterion does not stop short of a stationary point" begin
    prob = expfit_prob()
    ref = solve(
        prob, GaussNewton(); maxiters = 1000,
        termination_condition = AbsNormSafeBestTerminationMode(
            Base.Fix2(norm, 2); max_stalled_steps = 32
        )
    )
    got = solve(prob, GaussNewton(); maxiters = 1000)
    @test norm(expfit(got.u, nothing), 2) ≤ norm(expfit(ref.u, nothing), 2) * (1 + 1.0e-8)
    @test got.stats.nf ≤ ref.stats.nf
end

@testset "the same measurement is a failure on a square problem" begin
    # Freudenstein-Roth from a start that funnels into its classic local minimum: `JᵀF`
    # goes to zero while `F` does not, which is a local minimum of `‖F‖` and not a root.
    fr(u, p) = [
        -13.0 + u[1] + ((5.0 - u[2]) * u[2] - 2.0) * u[2],
        -29.0 + u[1] + ((u[2] + 1.0) * u[2] - 14.0) * u[2],
    ]
    frj(u, p) = [
        1.0 (10.0 - 3.0 * u[2]) * u[2] - 2.0
        1.0 (3.0 * u[2] + 2.0) * u[2] - 14.0
    ]
    prob = NonlinearProblem(NonlinearFunction{false}(fr; jac = frj), [15.0, -2.0], nothing)

    gtol = default_gradient_tolerance(Float64)
    on = AbsNormSafeBestTerminationMode(
        Base.Fix1(maximum, abs); max_stalled_steps = 32, gtol
    )
    sol = solve(prob, TrustRegion(); maxiters = 1000, termination_condition = on)

    @test maximum(abs, fr(sol.u, nothing)) > 1.0
    @test !SciMLBase.successful_retcode(sol.retcode)
    @test sol.retcode == ReturnCode.Stalled
end
