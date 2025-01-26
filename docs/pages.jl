# Put in a separate page so it can be used by SciMLDocs.jl

pages = [
    "index.md",
    "Getting Started with Nonlinear Rootfinding in Julia" => "tutorials/getting_started.md",
    "Tutorials" => Any[
        "tutorials/code_optimization.md",
        "tutorials/large_systems.md",
        "tutorials/modelingtoolkit.md",
        "tutorials/small_compile.md",
        "tutorials/iterator_interface.md",
        "tutorials/optimizing_parameterized_ode.md",
        "tutorials/snes_ex2.md"
    ],
    "Basics" => Any[
        "basics/nonlinear_problem.md",
        "basics/nonlinear_functions.md",
        "basics/solve.md",
        "basics/nonlinear_solution.md",
        "basics/autodiff.md",
        "basics/termination_condition.md",
        "basics/diagnostics_api.md",
        "basics/sparsity_detection.md",
        "basics/faq.md"
    ],
    "Solver Summaries and Recommendations" => Any[
        "solvers/nonlinear_system_solvers.md",
        "solvers/bracketing_solvers.md",
        "solvers/steady_state_solvers.md",
        "solvers/nonlinear_least_squares_solvers.md",
        "solvers/fixed_point_solvers.md"
    ],
    "Native Functionalities" => Any[
        "native/solvers.md",
        "native/simplenonlinearsolve.md",
        "native/bracketingnonlinearsolve.md",
        "native/steadystatediffeq.md",
        "native/descent.md",
        "native/globalization.md",
        "native/diagnostics.md"
    ],
    "Wrapped Solver APIs" => Any[
        "api/fastlevenbergmarquardt.md",
        "api/fixedpointacceleration.md",
        "api/leastsquaresoptim.md",
        "api/minpack.md",
        "api/nlsolve.md",
        "api/nlsolvers.md",
        "api/petsc.md",
        "api/siamfanlequations.md",
        "api/speedmapping.md",
        "api/sundials.md",
        "api/homotopycontinuation.md"
    ],
    "Sub-Packages" => Any[
        "api/SciMLJacobianOperators.md",
    ],
    "Development Documentation" => [
        "devdocs/internal_interfaces.md",
        "devdocs/linear_solve.md",
        "devdocs/jacobian.md",
        "devdocs/operators.md",
        "devdocs/algorithm_helpers.md"
    ],
    "Release Notes" => "release_notes.md",
    "References" => "references.md"
]
