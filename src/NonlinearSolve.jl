module NonlinearSolve

using ConcreteStructs: @concrete
using Reexport: @reexport
using PrecompileTools: @compile_workload, @setup_workload
using FastClosures: @closure

using ADTypes: ADTypes
using ArrayInterface: ArrayInterface
using CommonSolve: CommonSolve, init, solve, solve!
using DiffEqBase: DiffEqBase # Needed for `init` / `solve` dispatches
using LinearAlgebra: LinearAlgebra
using LineSearch: BackTracking
using NonlinearSolveBase: NonlinearSolveBase, AbstractNonlinearSolveAlgorithm,
                          NonlinearSolvePolyAlgorithm, pickchunksize

using SciMLBase: SciMLBase, ReturnCode, AbstractNonlinearProblem,
                 NonlinearFunction,
                 NonlinearProblem, NonlinearLeastSquaresProblem, NoSpecialize
using SymbolicIndexingInterface: SymbolicIndexingInterface
using StaticArraysCore: StaticArray

# Default Algorithm
using NonlinearSolveFirstOrder: NewtonRaphson, TrustRegion, LevenbergMarquardt, GaussNewton,
                                RUS, RobustMultiNewton
using NonlinearSolveQuasiNewton: Broyden, Klement
using SimpleNonlinearSolve: SimpleBroyden, SimpleKlement

# Default AD Support
using FiniteDiff: FiniteDiff          # Default Finite Difference Method
using ForwardDiff: ForwardDiff, Dual  # Default Forward Mode AD

# Sub-Packages that are re-exported by NonlinearSolve
using BracketingNonlinearSolve: BracketingNonlinearSolve
using LineSearch: LineSearch
using LinearSolve: LinearSolve
using NonlinearSolveFirstOrder: NonlinearSolveFirstOrder, GeneralizedFirstOrderAlgorithm
using NonlinearSolveQuasiNewton: NonlinearSolveQuasiNewton, QuasiNewtonAlgorithm
using NonlinearSolveSpectralMethods: NonlinearSolveSpectralMethods, GeneralizedDFSane
using SimpleNonlinearSolve: SimpleNonlinearSolve

const SII = SymbolicIndexingInterface

include("poly_algs.jl")
include("extension_algs.jl")

include("default.jl")

include("forward_diff.jl")

@setup_workload begin
    nonlinear_functions = (
        (NonlinearFunction{false, NoSpecialize}((u, p) -> u .* u .- p), 0.1),
        (NonlinearFunction{false, NoSpecialize}((u, p) -> u .* u .- p), [0.1]),
        (NonlinearFunction{true, NoSpecialize}((du, u, p) -> du .= u .* u .- p), [0.1])
    )

    nonlinear_problems = NonlinearProblem[]
    for (fn, u0) in nonlinear_functions
        push!(nonlinear_problems, NonlinearProblem(fn, u0, 2.0))
    end

    nonlinear_functions = (
        (NonlinearFunction{false, NoSpecialize}((u, p) -> (u .^ 2 .- p)[1:1]), [0.1, 0.0]),
        (
            NonlinearFunction{false, NoSpecialize}((
            u, p) -> vcat(u .* u .- p, u .* u .- p)),
            [0.1, 0.1]
        ),
        (
            NonlinearFunction{true, NoSpecialize}(
                (du, u, p) -> du[1] = u[1] * u[1] - p, resid_prototype = zeros(1)
            ),
            [0.1, 0.0]
        ),
        (
            NonlinearFunction{true, NoSpecialize}(
                (du, u, p) -> du .= vcat(u .* u .- p, u .* u .- p), resid_prototype = zeros(4)
            ),
            [0.1, 0.1]
        )
    )

    nlls_problems = NonlinearLeastSquaresProblem[]
    for (fn, u0) in nonlinear_functions
        push!(nlls_problems, NonlinearLeastSquaresProblem(fn, u0, 2.0))
    end

    @compile_workload begin
        @sync begin
            for prob in nonlinear_problems
                Threads.@spawn CommonSolve.solve(
                    prob, nothing; abstol = 1e-2, verbose = false
                )
            end

            for prob in nlls_problems
                Threads.@spawn CommonSolve.solve(
                    prob, nothing; abstol = 1e-2, verbose = false
                )
            end
        end
    end
end

# Rexexports
@reexport using SciMLBase, NonlinearSolveBase, LineSearch, ADTypes
@reexport using NonlinearSolveFirstOrder, NonlinearSolveSpectralMethods,
                NonlinearSolveQuasiNewton, SimpleNonlinearSolve, BracketingNonlinearSolve
@reexport using LinearSolve

# Poly Algorithms
export NonlinearSolvePolyAlgorithm, FastShortcutNonlinearPolyalg, FastShortcutNLLSPolyalg

# Extension Algorithms
export LeastSquaresOptimJL, FastLevenbergMarquardtJL, NLsolveJL, NLSolversJL,
       FixedPointAccelerationJL, SpeedMappingJL, SIAMFANLEquationsJL
export PETScSNES, CMINPACK

end
