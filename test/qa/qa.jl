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
        # Still non-public in their owning packages after the make-public round, across
        # the main module and the solver extensions:
        #   SciMLBase: AbstractSteadyStateProblem, __init, __solve
        #   NonlinearSolveBase(.Utils): Utils, evaluate_f, initialization_alg, nodual_value,
        #     safe_vec
        #   ForwardDiff: partials;  LeastSquaresOptim: Cholesky, LSMR, QR
        #   NonlinearSolveFirstOrder.RadiusUpdateSchemes: Bastin
        all_qualified_accesses_are_public = (;
            ignore = (
                :AbstractSteadyStateProblem, :__init, :__solve, :Utils, :evaluate_f,
                :initialization_alg, :nodual_value, :safe_vec, :partials, :Cholesky, :LSMR,
                :QR, :Bastin,
            ),
        ),
        # Still non-public in their owning packages after the make-public round:
        #   NonlinearSolveBase: AbstractNonlinearSolveAlgorithm, Utils, get_raw_f,
        #     is_fw_wrapped
        #   ForwardDiff: Dual;  StaticArraysCore: StaticArray
        #   NonlinearSolveFirstOrder: RUS;  NLsolve (re-export, owner NLSolversBase):
        #     NonDifferentiable
        all_explicit_imports_are_public = (;
            ignore = (
                :AbstractNonlinearSolveAlgorithm, :Utils, :get_raw_f, :is_fw_wrapped, :Dual,
                :StaticArray, :RUS, :NonDifferentiable,
            ),
        ),
    ),
)

# stale_deps / deps_compat are validated via the SimpleNonlinearSolve facade, which
# carries the SciMLJacobianOperators weak-dep ignore.
Aqua.test_stale_deps(SimpleNonlinearSolve; ignore = [:SciMLJacobianOperators])
Aqua.test_deps_compat(SimpleNonlinearSolve; ignore = [:SciMLJacobianOperators])
