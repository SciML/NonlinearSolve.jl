# Agent notes for NonlinearSolve.jl

Repository-specific conventions and pitfalls for automated contributors. Global operating
rules live outside this repository; keep this file to facts about this codebase.

## Validation before pushing

- Every sublibrary under `lib/` has its own project and test groups. Run
  `GROUP=Core julia --project=lib/<Sublib> -e 'using Pkg; Pkg.test()'` and `GROUP=QA` for
  each sublibrary whose sources you changed; the root `Pkg.test()` resolves the *released*
  sublibraries and does not test local `lib/` changes.
- Sublibrary `[sources]` entries are relative paths (`../NonlinearSolveBase`). Pkg resolves
  them against the *active* project, so scratch environments used to test a sublibrary
  must live under `lib/` (a sibling directory), never under `/tmp` or `~/tmp`.
- Never `Pkg.develop` into a sublibrary's own project: Pkg rewrites its `Project.toml`
  (`[sources]` with absolute paths, extra dependencies, reordered sections). Use a scratch
  project under `lib/` and develop the sublibraries into that.
- Format with Runic.jl (`julia -e 'using Runic; Runic.main(["--inplace", files...])'`), not
  the unrelated `runic` binary that may be on `PATH`, and run `typos`.
- When building commits from a working tree that predates recent `master` commits, merge
  `master` first. A diff against `origin/master` taken from an older base silently reverts
  the newer commits in every file it touches.

## Reactant support (`ReactantCore.@trace` in the solver loops)

The solver loops trace under `Reactant.@compile` by running the *ordinary* code path; the
helpers are in `lib/NonlinearSolveBase/src/reactant.jl`.

- `ReactantCore.within_compile()` is a compile-time `false` outside Reactant, so a helper
  that checks it costs nothing on the host. Call it only from ordinary functions: inside a
  `ReactantCore.@trace while`/`if` body it returns `false` even while tracing, because the
  macro captures every symbol of the body, the module included, as a loop variable. Loop
  bodies call self-gating helpers (`dealias_traced!`, `Utils.fresh`, ...) unconditionally.
- Solver caches are `@concrete`, so scalar loop state (`nsteps`, `force_stop`, `retcode`,
  trust-region counters) must be traced at construction with `maybe_traced`; promoting a
  field after the cache exists fails with a type error.
- Reactant records one path per traced object among loop-carried values and requires the
  same set after each iteration. Aliases between cache fields break this; `dealias_traced!`
  refreshes every traced leaf at the loop boundary. `@bb copyto!(dst, src)` and `@bb copy`
  rebind (`dst = src`) for traced arrays, so they create aliases.
- Write decisions on traced values with `ifelse`/`select`, not `if`. A `@trace if` that
  mutates the cache is avoided in the loop path; where a `@trace if` is used, its branches
  must assign only loop-state leaves (a whole solution object as a branch output has to be
  materialized for the untaken side), and no variable in scope may be named `args`, which
  the macro uses for its captured-variable bundle.
- Under compilation the returned `NonlinearSolution` has `stats === nothing` (`NLStats`
  holds `Int`s) and `prob === nothing` (a problem's `Base.Pairs` keyword arguments cannot
  be rebuilt by Reactant's result codegen); the default termination mode is
  `AbsNormTerminationMode`, the Jacobian-reuse policy is switched off, and initialization
  failure cannot be reported.
- Not traceable at present, with the reason recorded next to the Reactant test matrix in
  `test/Reactant/reactant_tests.jl`: line-search globalization (`norm(x, Inf)` scalar-indexes
  under Reactant), `DFSane`, and trust-region schemes needing a reverse-mode
  vector-Jacobian product (`RobustMultiNewton`).
- The Reactant test group pins unreleased branches of Reactant, SciMLBase and
  DifferentiationInterface in `test/runtests.jl`; update the pins when those release.
