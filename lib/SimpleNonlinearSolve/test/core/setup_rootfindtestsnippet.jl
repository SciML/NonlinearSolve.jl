using StaticArrays, Random, LinearAlgebra, ForwardDiff, NonlinearSolveBase, SciMLBase
using ADTypes, PolyesterForwardDiff, ReverseDiff, LineSearch

# Conditionally import Enzyme only if not on Julia prerelease
if isempty(VERSION.prerelease) && VERSION < v"1.12"
    using Enzyme
end

quadratic_f(u, p) = u .* u .- p
quadratic_f!(du, u, p) = (du .= u .* u .- p)

function newton_fails(u, p)
    return 0.010000000000000002 .+
        10.000000000000002 ./ (
        1 .+
            (
            0.21640425613334457 .+
                216.40425613334457 ./ (
                1 .+
                    (
                    0.21640425613334457 .+
                        216.40425613334457 ./ (1 .+ 0.0006250000000000001(u .^ 2.0))
                ) .^ 2.0
            )
        ) .^
            2.0
    ) .- 0.0011552453009332421u .- p
end

const TERMINATION_CONDITIONS = [
    NormTerminationMode(Base.Fix1(maximum, abs)),
    RelTerminationMode(),
    RelNormTerminationMode(Base.Fix1(maximum, abs)),
    RelNormSafeTerminationMode(Base.Fix1(maximum, abs)),
    RelNormSafeBestTerminationMode(Base.Fix1(maximum, abs)),
    AbsTerminationMode(),
    AbsNormTerminationMode(Base.Fix1(maximum, abs)),
    AbsNormSafeTerminationMode(Base.Fix1(maximum, abs)),
    AbsNormSafeBestTerminationMode(Base.Fix1(maximum, abs)),
]

function run_nlsolve_oop(f::F, u0, p = 2.0; solver, broken_inferred = false) where {F}
    prob = NonlinearProblem{false}(f, u0, p)
    @test @inferred(solve(prob, solver; abstol = 1.0e-9)) isa
        SciMLBase.AbstractNonlinearSolution broken = broken_inferred
    return solve(prob, solver; abstol = 1.0e-9)
end
function run_nlsolve_iip(f!::F, u0, p = 2.0; solver, broken_inferred = false) where {F}
    prob = NonlinearProblem{true}(f!, u0, p)
    @test @inferred(solve(prob, solver; abstol = 1.0e-9)) isa
        SciMLBase.AbstractNonlinearSolution broken = broken_inferred
    return solve(prob, solver; abstol = 1.0e-9)
end
