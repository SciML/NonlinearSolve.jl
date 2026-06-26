using SciMLTesting, NonlinearSolve, SimpleNonlinearSolve, SciMLBase, Aqua, Test
# Load the wrapper packages so NonlinearSolve's solver extensions are present for
# ExplicitImports to analyze (matches the pre-run_qa explicit_imports.jl set).
using ADTypes
import FastLevenbergMarquardt, FixedPointAcceleration, LeastSquaresOptim, MINPACK,
    NLsolve, NLSolvers, SIAMFANLEquations, SpeedMapping

run_qa(
    NonlinearSolve;
    explicit_imports = true,
    aqua_kwargs = (;
        # stale_deps / deps_compat are checked on the SimpleNonlinearSolve facade
        # below (with the SciMLJacobianOperators ignore); persistent_tasks stays off
        # for the umbrella package.
        stale_deps = false,
        deps_compat = false,
        persistent_tasks = false,
        ambiguities = (; recursive = false),
        piracies = (;
            treat_as_own = [
                NonlinearProblem, NonlinearLeastSquaresProblem,
                SciMLBase.AbstractNonlinearProblem,
                SimpleNonlinearSolve.AbstractSimpleNonlinearSolveAlgorithm,
            ],
        ),
    ),
    ei_kwargs = (;
        # NonDifferentiable is owned by NLSolversBase and re-exported through NLsolve
        # (where the NLsolve extension imports it from).
        all_explicit_imports_via_owners = (; ignore = (:NonDifferentiable,)),
        # Still non-public in their owning packages (the public-API round covered only
        # SciMLBase/CommonSolve/ArrayInterface), across the main module and the solver
        # extensions (FastLevenbergMarquardt, FixedPointAcceleration, LeastSquaresOptim,
        # MINPACK, NLsolve, NLSolvers, SIAMFANLEquations, SpeedMapping):
        #   SciMLBase: AbstractSteadyStateProblem, NonlinearAliasSpecifier, __init, __solve
        #   NonlinearSolveBase(.Utils): AbstractNonlinearSolveAlgorithm, Utils, evaluate_f,
        #     initialization_alg, nodual_value, safe_vec
        #   ForwardDiff: Dual, partials;  StaticArraysCore: StaticArray
        #   LinearSolve: LSMR, QR;  LinearAlgebra: Cholesky
        #   SIAMFANLEquations: Bastin;  SpeedMapping: RUS
        all_qualified_accesses_are_public = (;
            ignore = (
                :AbstractSteadyStateProblem, :NonlinearAliasSpecifier, :__init, :__solve,
                :AbstractNonlinearSolveAlgorithm, :Utils, :evaluate_f, :initialization_alg,
                :nodual_value, :safe_vec, :Dual, :partials, :StaticArray, :LSMR, :QR,
                :Cholesky, :Bastin, :RUS,
            ),
        ),
        # Still non-public in their owning packages, main module + solver extensions:
        #   NonlinearSolveBase(.Utils): get_raw_f, is_fw_wrapped,
        #     AbstractNonlinearSolveAlgorithm, Utils, safe_vec
        #   ForwardDiff: Dual;  StaticArraysCore: StaticArray;  SpeedMapping: RUS
        #   NLsolve (re-export, owner NLSolversBase): NonDifferentiable
        all_explicit_imports_are_public = (;
            ignore = (
                :get_raw_f, :is_fw_wrapped, :AbstractNonlinearSolveAlgorithm, :Utils,
                :safe_vec, :Dual, :StaticArray, :RUS, :NonDifferentiable,
            ),
        ),
    ),
)

# stale_deps / deps_compat are validated via the SimpleNonlinearSolve facade, which
# carries the SciMLJacobianOperators weak-dep ignore.
Aqua.test_stale_deps(SimpleNonlinearSolve; ignore = [:SciMLJacobianOperators])
Aqua.test_deps_compat(SimpleNonlinearSolve; ignore = [:SciMLJacobianOperators])
