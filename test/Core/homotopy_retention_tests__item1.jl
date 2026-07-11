# Driver-level inner-algorithm retention: the homotopy continuation drivers reuse one
# inner cache across warm-started steps and arm best-subalgorithm retention on their
# tracking reinit!s, so a polyalgorithm inner (the default) no longer re-fails its
# cheap rungs — nor pays a full-ladder reinit! — on every continuation step.
using NonlinearSolve, SciMLBase

using Test

const NF = Ref(0)

function count_nf(prob, alg)
    NF[] = 0
    sol = solve(prob, alg)
    return sol, NF[]
end

# n = 50 coupled cubic with u = ones(n) solving λ = 1: the post-#1029 benchmark
# problem where the default inner used to cost ~1.65x a plain NewtonRaphson inner
# (2295 vs 1400 residual calls; 1577 with retention).
const N = 50
function coupled_cubic!(du, u, p, λ)
    NF[] += 1
    n = length(u)
    for i in 1:n
        c = 2.0 + 0.25 * (i > 1) + 0.25 * (i < n)
        du[i] = u[i] + 0.25 * (i > 1 ? u[i - 1] : 0.0) +
            0.25 * (i < n ? u[i + 1] : 0.0) + λ * u[i]^3 - c
    end
    return nothing
end

@testset "sweep: default inner stays close to a Newton inner (n = 50)" begin
    prob = HomotopyProblem{true}(coupled_cubic!, ones(N), nothing; λspan = (0.0, 1.0))
    sol_d, nf_d = count_nf(prob, HomotopySweep())
    sol_n, nf_n = count_nf(prob, HomotopySweep(; inner = NewtonRaphson()))
    @test SciMLBase.successful_retcode(sol_d)
    @test SciMLBase.successful_retcode(sol_n)
    @test sol_d.u ≈ ones(N) atol = 1.0e-6
    # measured 1577 vs 1400 (1.13x) with retention, 2295 (1.64x) without; the
    # generous margin only trips when retention regresses
    @test nf_d ≤ 1.35 * nf_n
end

@testset "sweep: default inner stays close to a Newton inner (scalar)" begin
    H(u, p, λ) = (NF[] += 1; [(1 - λ) * (u[1] - 4.0) + λ * (u[1]^2 - 4.0)])
    prob = HomotopyProblem(H, [4.0], nothing; λspan = (0.0, 1.0))
    sol_d, nf_d = count_nf(prob, HomotopySweep())
    sol_n, nf_n = count_nf(prob, HomotopySweep(; inner = NewtonRaphson()))
    @test SciMLBase.successful_retcode(sol_d)
    @test SciMLBase.successful_retcode(sol_n)
    @test sol_d.u[1] ≈ 2.0 atol = 1.0e-6
    # measured 94 vs 86 (1.09x) with retention, 150 (1.74x) without
    @test nf_d ≤ 1.3 * nf_n
end

@testset "arclength: retention on the augmented corrector cache" begin
    H(u, p, λ) = (NF[] += 1; [u[1]^3 - 3u[1] - (-3 + 6λ)])
    prob = HomotopyProblem(H, [-2.1038034], nothing; λspan = (0.0, 1.0))
    sol_d, nf_d = count_nf(prob, ArcLengthContinuation())
    sol_n, nf_n = count_nf(prob, ArcLengthContinuation(; inner = NewtonRaphson()))
    @test SciMLBase.successful_retcode(sol_d)
    @test SciMLBase.successful_retcode(sol_n)
    @test sol_d.u[1] ≈ sol_n.u[1] atol = 1.0e-6
    # measured 267 vs 188 (1.42x) with retention, 381 (2.03x) without; the corrector
    # fails repeatedly while rounding the fold, and each polyalgorithm failure runs
    # its whole ladder, so parity with a bare Newton inner is not reachable here
    @test nf_d ≤ 1.7 * nf_n
end

@testset "anchor still runs the full ladder" begin
    # The λspan[1] anchor is the one cold-start solve: Newton stalls from u0 = 0
    # (singular Jacobian at the origin), so the anchor only succeeds if the inner
    # ladder escalates to Broyden there. Retention on the interior steps then starts
    # from the anchor's winner.
    H(u, p, λ) = [u[1]^3 - 2.0 - λ]
    prob = HomotopyProblem(H, [0.0], nothing; λspan = (0.0, 1.0))
    inner = NonlinearSolvePolyAlgorithm((NewtonRaphson(), Broyden()))
    sol = solve(prob, HomotopySweep(; inner))
    @test SciMLBase.successful_retcode(sol)
    @test sol.u[1] ≈ cbrt(3.0) atol = 1.0e-6
end

@testset "sticky-rung failure mid-continuation escalates and the sweep still lands" begin
    # Difficulty jump: past λ = 0.5 the residual reports NaN to any non-Float64
    # (dual) evaluation, so the retained Jacobian-based rung (NewtonRaphson) fails
    # every corrector there while the identity-initialized derivative-free Broyden
    # rung still converges — the sweep only lands if escalation keeps working on
    # retained steps. The path u = 2 + λ itself stays smooth and easy.
    function gated(u, p, λ)
        bad = λ > 0.5 && !(u[1] isa Float64)
        r = u[1] - 2.0 - λ
        return [bad ? oftype(r, NaN) : r]
    end
    prob = HomotopyProblem(gated, [2.0], nothing; λspan = (0.0, 1.0))
    inner = NonlinearSolvePolyAlgorithm((NewtonRaphson(), Broyden()))
    sol = solve(prob, HomotopySweep(; inner))
    @test SciMLBase.successful_retcode(sol)
    @test sol.u[1] ≈ 3.0 atol = 1.0e-6
end
