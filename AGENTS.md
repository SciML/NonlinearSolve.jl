# Solver implementation conventions

- Reuse `NonlinearSolveBase.construct_jacobian_cache` and the existing descent or linear-solver caches for new algorithms. Honor `linsolve`, `concrete_jac`, sparse `jac_prototype`, JVP/VJP callbacks and AD backends, `linsolve_kwargs`, and cache reinitialization. Do not unconditionally convert Jacobians to dense matrices or bypass the selected linear solver with a direct factorization.
- Test representation as well as convergence: sparse prototypes must remain sparse, and Krylov paths must work through operators without materializing the Jacobian. Exercise parameter-changing reinitialization and preconditioning on these paths.

- Share bound-handling and globalization primitives across bounded algorithms, including `BoundedTrustRegion`. Preserve each method's step geometry and radius policy when consolidating helpers. Line-search algorithms, including their bound-aware trial and acceptance logic, belong in LineSearch.jl; solver code should reuse its public caches and results instead of implementing local backtracking loops.
