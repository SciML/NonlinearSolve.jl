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
        # Non-public names qualified-accessed from their owning packages (solver exts):
        #   SciMLBase: NLStats, NonlinearAliasSpecifier, __solve, build_solution
        #   NonlinearSolveBase(.Utils): Utils, evaluate_f
        all_qualified_accesses_are_public = (;
            ignore = (
                :NLStats, :NonlinearAliasSpecifier, :Utils, :__solve, :build_solution,
                :evaluate_f,
            ),
        ),
        # Non-public names explicitly imported from NonlinearSolveBase (solver exts):
        all_explicit_imports_are_public = (; ignore = (:get_raw_f, :is_fw_wrapped)),
    ),
)

# stale_deps / deps_compat are validated via the SimpleNonlinearSolve facade, which
# carries the SciMLJacobianOperators weak-dep ignore.
Aqua.test_stale_deps(SimpleNonlinearSolve; ignore = [:SciMLJacobianOperators])
Aqua.test_deps_compat(SimpleNonlinearSolve; ignore = [:SciMLJacobianOperators])
