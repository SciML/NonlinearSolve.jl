using NonlinearSolveBase, SciMLBase, LinearAlgebra, Test
using NonlinearSolveBase: AbsNormSafeBestTerminationMode, AbsNormSafeTerminationMode,
    AbsNormTerminationMode, RelNormSafeBestTerminationMode, RelNormSafeTerminationMode,
    check_gradient_and_update!, default_gradient_tolerance, gradient_measure_supported,
    gradient_stationarity_measure
using SciMLBase: NonlinearLeastSquaresProblem, NonlinearProblem, ReturnCode
using StaticArrays: SA

@static if VERSION ≥ v"1.11"
    @testset "new names are public" begin
        for name in (
                :check_gradient_and_update!, :default_gradient_tolerance,
                :gradient_measure_supported, :gradient_stationarity_measure,
            )
            @test Base.ispublic(NonlinearSolveBase, name)
        end
    end
end

@testset "measure is the cosine between the residual and each Jacobian column" begin
    # A residual orthogonal to the only column is stationary regardless of its size.
    @test gradient_stationarity_measure([1.0; 0.0;;], [0.0, 3.0]) == 0.0
    # A residual parallel to the column is maximally non-stationary.
    @test gradient_stationarity_measure([1.0; 0.0;;], [2.0, 0.0]) == 1.0
    @test gradient_stationarity_measure(2.0, 3.0) == 1.0
    # The largest column cosine wins.
    J = [1.0 0.0; 0.0 1.0]
    @test gradient_stationarity_measure(J, [3.0, 4.0]) ≈ 0.8
end

@testset "scale invariance" begin
    # A criterion on `‖JᵀF‖` alone moves under both rescalings; the cosine does not.
    J = [1.0 4.0; 2.0 -1.0; -3.0 0.5]
    F = [0.3, -1.2, 2.0]
    base = gradient_stationarity_measure(J, F)

    for s in ([1.0e6, 1.0e-6], [3.0, 3.0], [1.0e-8, 1.0])
        S = Diagonal(s)
        @test gradient_stationarity_measure(J * inv(S), F) ≈ base rtol = 1.0e-12
    end
    for c in (1.0e8, 1.0e-8, -5.0)
        @test gradient_stationarity_measure(c * J, c * F) ≈ base rtol = 1.0e-12
    end
    # An orthogonal transformation of the residual leaves the least-squares problem, and
    # therefore the measure, unchanged.
    θ = 0.7
    Q = [
        cos(θ) -sin(θ) 0.0
        sin(θ) cos(θ) 0.0
        0.0 0.0 1.0
    ]
    @test gradient_stationarity_measure(Q * J, Q * F) ≈ base rtol = 1.0e-12

    # The unnormalised gradient is not invariant, which is why it is not the criterion.
    @test !isapprox(
        norm((J * inv(Diagonal([1.0e6, 1.0e-6])))'F, Inf), norm(J'F, Inf); rtol = 1.0e-3
    )
end

@testset "measure is undefined rather than wrong on degenerate input" begin
    J = [1.0 0.0; 0.0 1.0]
    @test gradient_stationarity_measure(J, [0.0, 0.0]) === nothing
    @test gradient_stationarity_measure(J, [NaN, 1.0]) === nothing
    @test gradient_stationarity_measure(J, [Inf, 1.0]) === nothing
    @test gradient_stationarity_measure(0.0, 1.0) === nothing
    # A non-finite Jacobian leaves a `NaN` that every caller must treat as "do not fire".
    @test isnan(gradient_stationarity_measure([NaN 0.0; 0.0 1.0], [1.0, 2.0]))
    # Shape mismatch is not a silent zero.
    @test gradient_stationarity_measure(J, [1.0, 2.0, 3.0]) === nothing
end

@testset "gradient_measure_supported" begin
    @test gradient_measure_supported([1.0 0.0; 0.0 1.0])
    @test gradient_measure_supported(Diagonal([1.0, 2.0]))
    @test gradient_measure_supported(1.0)
    # A matrix-free operator has no column norms, so the criterion must stand down.
    @test !gradient_measure_supported(nothing)
    @test !gradient_measure_supported(I)
end

@testset "gtol defaults to off on every mode" begin
    for mode in (
            AbsNormSafeTerminationMode(Base.Fix2(norm, 2)),
            AbsNormSafeBestTerminationMode(Base.Fix2(norm, 2)),
            RelNormSafeTerminationMode(Base.Fix2(norm, 2)),
            RelNormSafeBestTerminationMode(Base.Fix2(norm, 2)),
        )
        @test NonlinearSolveBase.gradient_tolerance(mode) === nothing
    end
    @test NonlinearSolveBase.gradient_tolerance(AbsNormTerminationMode(Base.Fix2(norm, 2))) ===
        nothing
    @test NonlinearSolveBase.gradient_tolerance(
        AbsNormSafeBestTerminationMode(Base.Fix2(norm, 2); gtol = 1.0e-6)
    ) == 1.0e-6

    @test default_gradient_tolerance(Float64) == sqrt(eps(Float64))
    @test default_gradient_tolerance(Float32) == sqrt(eps(Float32))
end

@testset "least-squares default opts in, square default does not" begin
    ls = NonlinearSolveBase.default_termination_mode(
        NonlinearLeastSquaresProblem((u, p) -> [u[1] - 1.0, 2.0], [0.0]), Val(:regular)
    )
    @test ls.gtol == default_gradient_tolerance(Float64)

    sq = NonlinearSolveBase.default_termination_mode(
        NonlinearProblem((u, p) -> u, [1.0]), Val(:regular)
    )
    @test NonlinearSolveBase.gradient_tolerance(sq) === nothing

    # `:simple` (SimpleNonlinearSolve) is untouched.
    for prob in (
            NonlinearLeastSquaresProblem((u, p) -> [u[1] - 1.0, 2.0], [0.0]),
            NonlinearProblem((u, p) -> u, [1.0]),
        )
        @test NonlinearSolveBase.gradient_tolerance(
            NonlinearSolveBase.default_termination_mode(prob, Val(:simple))
        ) === nothing
    end
end

# A minimal stand-in for a solver cache: `check_gradient_and_update!` reads only these
# fields, which is what lets a solver package opt in without a new cache type.
mutable struct FakeSolverCache{C}
    termination_cache::C
    retcode::ReturnCode.T
    force_stop::Bool
end

function fake_cache(prob, mode, du, u)
    tc = SciMLBase.init(prob, mode, du, u; abstol = 1.0e-8, reltol = 1.0e-8)
    return FakeSolverCache(tc, ReturnCode.Default, false)
end

NonlinearSolveBase.get_u(c::FakeSolverCache) = c.termination_cache.u

@testset "verdict flips between least-squares and square problems" begin
    du = [0.0, 3.0]
    u = [1.0]
    J = [1.0; 0.0;;]              # residual orthogonal to the only column: stationary
    mode() = AbsNormSafeBestTerminationMode(Base.Fix2(norm, 2); gtol = 1.0e-8)

    lsq = fake_cache(
        NonlinearLeastSquaresProblem((x, p) -> du, u), mode(), du, u
    )
    @test check_gradient_and_update!(lsq, J, du, u)
    @test lsq.force_stop
    @test lsq.retcode == ReturnCode.Success
    @test SciMLBase.successful_retcode(lsq.retcode)

    sq = fake_cache(NonlinearProblem((x, p) -> du, u), mode(), du, u)
    @test check_gradient_and_update!(sq, J, du, u)
    @test sq.force_stop
    # Stationary with a residual this large is a local minimum, not a root.
    @test sq.retcode == ReturnCode.Stalled
    @test !SciMLBase.successful_retcode(sq.retcode)
end

@testset "does not fire when it should not" begin
    u = [1.0]
    mode() = AbsNormSafeBestTerminationMode(Base.Fix2(norm, 2); gtol = 1.0e-8)
    prob(du) = NonlinearLeastSquaresProblem((x, p) -> du, u)

    # Non-stationary.
    du = [2.0, 3.0]
    c = fake_cache(prob(du), mode(), du, u)
    @test !check_gradient_and_update!(c, [1.0; 0.0;;], du, u)
    @test !c.force_stop

    # `gtol` unset: the criterion is inert, which is what keeps every existing mode and
    # every user-defined mode behaving exactly as before.
    du = [0.0, 3.0]
    c = fake_cache(prob(du), AbsNormSafeBestTerminationMode(Base.Fix2(norm, 2)), du, u)
    @test !check_gradient_and_update!(c, [1.0; 0.0;;], du, u)
    @test !c.force_stop

    # No usable Jacobian representation: degrade, never error.
    du = [0.0, 3.0]
    c = fake_cache(prob(du), mode(), du, u)
    @test !check_gradient_and_update!(c, nothing, du, u)
    @test !c.force_stop

    # Non-finite Jacobian makes the measure `NaN`; terminating on it would report a
    # stationary point that was never established.
    du = [1.0, 2.0]
    c = fake_cache(prob(du), mode(), du, u)
    @test !check_gradient_and_update!(c, [NaN; 0.0;;], du, u)
    @test !c.force_stop

    # Non-finite residual, likewise.
    du = [Inf, -Inf]
    c = fake_cache(prob(du), mode(), du, u)
    @test !check_gradient_and_update!(c, [1.0; 0.0;;], du, u)
    @test !c.force_stop
end

@testset "a Best mode fires only at the iterate it reports" begin
    # The mode retains the lowest-residual iterate. If the current one is worse, firing
    # would attach a stationarity verdict to a point that was never tested.
    u = [1.0]
    du_best = [0.0, 1.0]
    mode = AbsNormSafeBestTerminationMode(Base.Fix2(norm, 2); gtol = 1.0e-8)
    c = fake_cache(NonlinearLeastSquaresProblem((x, p) -> du_best, u), mode, du_best, u)

    du_worse = [0.0, 5.0]      # stationary, but a worse residual than the retained best
    @test !check_gradient_and_update!(c, [1.0; 0.0;;], du_worse, u)
    @test !c.force_stop
    # At the retained best it fires.
    @test check_gradient_and_update!(c, [1.0; 0.0;;], du_best, u)
end

@testset "a user-defined mode keeps working and never reaches the gradient path" begin
    struct UserGradTestMode <: NonlinearSolveBase.AbstractNonlinearTerminationMode end
    function NonlinearSolveBase.check_convergence(
            ::UserGradTestMode, duₙ, uₙ, _, abstol, __
        )
        return maximum(abs, duₙ) ≤ abstol
    end

    prob = NonlinearLeastSquaresProblem((u, p) -> [0.0, 3.0], [1.0])
    tc = SciMLBase.init(
        prob, UserGradTestMode(), [0.0, 3.0], [1.0]; abstol = 1.0e-8, reltol = 1.0e-8
    )
    @test NonlinearSolveBase.gradient_tolerance(UserGradTestMode()) === nothing

    c = FakeSolverCache(tc, ReturnCode.Default, false)
    @test !check_gradient_and_update!(c, [1.0; 0.0;;], [0.0, 3.0], [1.0])
    @test !c.force_stop

    # The documented call signature still decides termination.
    @test !tc([1.0, 1.0], [1.0], [1.0])
    @test tc([1.0e-12, 1.0e-12], [1.0], [1.0])
end

@testset "static and scalar states" begin
    @test gradient_stationarity_measure(SA[1.0 0.0; 0.0 1.0], SA[3.0, 4.0]) ≈ 0.8
    @test gradient_stationarity_measure(2.0, 0.0) === nothing
end
