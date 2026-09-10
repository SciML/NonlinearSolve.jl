module NonlinearSolve

using ConcreteStructs: @concrete
using Reexport: @reexport
using PrecompileTools: @compile_workload, @setup_workload

using ADTypes: ADTypes
using ArrayInterface: ArrayInterface
using CommonSolve: CommonSolve, init, solve, solve!
using LinearAlgebra: LinearAlgebra
using LineSearch: BackTracking
using NonlinearSolveBase: NonlinearSolveBase, AbstractNonlinearSolveAlgorithm,
    NonlinearSolvePolyAlgorithm, HomotopyPolyAlgorithm, pickchunksize, NonlinearVerbosity

using SciMLBase: SciMLBase, ReturnCode, AbstractNonlinearProblem,
    NonlinearFunction,
    NonlinearProblem, NonlinearLeastSquaresProblem
using SymbolicIndexingInterface: SymbolicIndexingInterface
using StaticArraysCore: StaticArray

# Default Algorithm
using NonlinearSolveFirstOrder: BoundedTrustRegion, NewtonRaphson, TrustRegion,
    LevenbergMarquardt, GaussNewton, RUS, RobustMultiNewton
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
        (NonlinearFunction{false}((u, p) -> u .* u .- p), 0.1),
        (NonlinearFunction{false}((u, p) -> u .* u .- p), [0.1]),
        (NonlinearFunction{true}((du, u, p) -> du .= u .* u .- p), [0.1]),
    )

    nonlinear_problems = NonlinearProblem[]
    for (fn, u0) in nonlinear_functions
        push!(nonlinear_problems, NonlinearProblem(fn, u0, 2.0))
    end

    # IIP with Vector{Float64} params
    push!(
        nonlinear_problems,
        NonlinearProblem(
            NonlinearFunction{true}((du, u, p) -> du .= u .* u .- p),
            [0.1],
            [2.0],
        ),
    )

    # IIP with NullParameters (no p)
    push!(
        nonlinear_problems,
        NonlinearProblem(
            NonlinearFunction{true}((du, u, p) -> du .= u .* u .- 2.0),
            [0.1],
        ),
    )

    nonlinear_functions = (
        (NonlinearFunction{false}((u, p) -> (u .^ 2 .- p)[1:1]), [0.1, 0.0]),
        (
            NonlinearFunction{false}(
                (u, p) -> vcat(u .* u .- p, u .* u .- p)
            ),
            [0.1, 0.1],
        ),
        (
            NonlinearFunction{true}(
                (du, u, p) -> du[1] = u[1] * u[1] - p, resid_prototype = zeros(1)
            ),
            [0.1, 0.0],
        ),
        (
            NonlinearFunction{true}(
                (du, u, p) -> du .= vcat(u .* u .- p, u .* u .- p),
                resid_prototype = zeros(4)
            ),
            [0.1, 0.1],
        ),
    )

    nlls_problems = NonlinearLeastSquaresProblem[]
    for (fn, u0) in nonlinear_functions
        push!(nlls_problems, NonlinearLeastSquaresProblem(fn, u0, 2.0))
    end

    # NLLS with Vector{Float64} params
    push!(
        nlls_problems,
        NonlinearLeastSquaresProblem(
            NonlinearFunction{true}(
                (du, u, p) -> du .= vcat(u .* u .- p, u .* u .- p),
                resid_prototype = zeros(4),
            ),
            [0.1, 0.1],
            [2.0, 2.0],
        ),
    )

    # AutoDePSpecialize opaque-p path. Both containers are covered: an isbits `p`
    # packs into an `OpaqueParams` and a non-isbits `p` into an `OpaqueRef`, each
    # a single wrapped-residual signature shared across every parameter type of
    # its kind.
    push!(
        nonlinear_problems,
        NonlinearProblem(
            NonlinearFunction{true, SciMLBase.AutoDePSpecialize}(
                (du, u, p) -> (du .= u .* u .- p.a)
            ),
            [0.1],
            (a = 2.0,),
        ),
    )
    push!(
        nonlinear_problems,
        NonlinearProblem(
            NonlinearFunction{true, SciMLBase.AutoDePSpecialize}(
                (du, u, p) -> (du .= u .* u .- p[1])
            ),
            [0.1],
            [2.0],
        ),
    )

    push!(
        nonlinear_problems,
        NonlinearProblem(
            NonlinearFunction{true, SciMLBase.AutoDespecialize}(
                (du, u, p) -> (du .= u .* u .- p.a)
            ),
            [0.1],
            (a = 2.0,),
        ),
    )

    nlp_algs = [NewtonRaphson(), TrustRegion(), BoundedTrustRegion(), LevenbergMarquardt()]
    nlls_algs = [GaussNewton(), TrustRegion(), BoundedTrustRegion(), LevenbergMarquardt()]

    @compile_workload begin
        @sync begin
            for prob in nonlinear_problems, alg in nlp_algs
                Threads.@spawn CommonSolve.solve(prob, alg; abstol = 1.0e-2, verbose = NonlinearVerbosity())
            end

            for prob in nlls_problems, alg in nlls_algs
                Threads.@spawn CommonSolve.solve(prob, alg; abstol = 1.0e-2, verbose = NonlinearVerbosity())
            end

            # Default algorithms — the paths hit by solve(prob) with no algorithm
            # NonlinearProblem → FastShortcutNonlinearPolyalg
            # NonlinearLeastSquaresProblem → FastShortcutNLLSPolyalg
            for prob in nonlinear_problems
                Threads.@spawn CommonSolve.solve(prob; abstol = 1.0e-2, verbose = NonlinearVerbosity())
            end
            for prob in nlls_problems
                Threads.@spawn CommonSolve.solve(prob; abstol = 1.0e-2, verbose = NonlinearVerbosity())
            end

            # `solve(prob)` with no keyword arguments is a distinct
            # specialization from the kwarg-carrying calls above (the keyword
            # NamedTuple's type participates), and it is the form most user code
            # writes, so every problem gets a bare sweep too.
            for prob in nonlinear_problems
                Threads.@spawn CommonSolve.solve(prob)
            end
            for prob in nlls_problems
                Threads.@spawn CommonSolve.solve(prob)
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
export NonlinearSolvePolyAlgorithm, FastShortcutNonlinearPolyalg, FastShortcutHomotopyPolyalg

# Extension Algorithms
export LeastSquaresOptimJL, FastLevenbergMarquardtJL, NLsolveJL, NLSolversJL,
    FixedPointAccelerationJL, SpeedMappingJL, SIAMFANLEquationsJL
export PETScSNES, CMINPACK

end
