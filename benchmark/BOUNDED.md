# Bounded solver benchmarks

This suite measures local solvers for box-constrained roots and nonlinear least squares.
It contains 79 problem/start/representation combinations across 12 families. The boxes
and starts are adaptations for this comparison, not a claim to reproduce a canonical
bounded benchmark suite.

The 57 development cases cover Rosenbrock, Powell singular, Beale, Brown badly scaled,
exponential curve fitting, rank-deficient and underdetermined linear systems, fixed
variables, and nonlinear diffusion. The 22 additional cases cover Freudenstein–Roth,
Wood, Bard, Kowalik–Osborne, and automatic-Jacobian versions of selected development
cases. Diffusion sizes are 64, 256, and 1,024, with sparse and matrix-free representations,
interior roots and manufactured active-bound stationary points, and two starts.

The classical residual definitions and data can be checked against the
[MINPACK least-squares test functions](https://www.netlib.org/minpack/ex/file17).
`bounded_reference_costs.py` independently computes the two nonzero fitting reference
costs using [SciPy least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html).
`validate_bounded_problems.jl` checks every analytic benchmark Jacobian against
ForwardDiff. The remaining reference costs follow from exact roots, orthogonal
projection for fixed variables, or the manufactured diffusion stationarity equations.

## Reproduction

Run from the repository root. Use a local environment on a drive with sufficient space:

```sh
export TMPDIR="$PWD/.benchmark/tmp"
mkdir -p "$TMPDIR"
export JULIA_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0
julia benchmark/setup_bounded.jl
julia --project=.benchmark/bounded benchmark/validate_bounded_problems.jl
julia --project=.benchmark/bounded benchmark/bounded_solvers.jl .benchmark/development.csv
julia --project=.benchmark/bounded benchmark/bounded_solvers.jl .benchmark/validation.csv '' 5 '' validation
julia --project=.benchmark/bounded benchmark/bounded_cache.jl .benchmark/cache.csv
julia --project=.benchmark/bounded benchmark/bounded_pairwise.jl .benchmark/pairwise.csv
python3 benchmark/analyze_bounded.py .benchmark/summary.json .benchmark/development.csv .benchmark/validation.csv
python3 benchmark/bounded_reference_costs.py
python3 benchmark/plot_bounded.py .benchmark/performance.svg .benchmark/development.csv .benchmark/validation.csv
```

The setup script explicitly develops the checkout and its local sublibraries into the
benchmark environment. The reference-cost script requires NumPy and SciPy; the analysis script uses only Python's
standard library; plotting requires NumPy and Matplotlib. `bounded_solvers.jl` also accepts a case-name filter, requested sample
count, and algorithm-name filter. Algorithm filters `native` and `baseline` select the
seven individual native configurations and the three transformed baselines respectively.

## Measurements and acceptance

Each configuration is solved once before timing. BenchmarkTools then records the median
of up to five single-evaluation samples with a 60-second sampling budget. The CSV records
how many samples completed, allocations, allocated bytes, the return code, and independent
accuracy checks. Compilation is excluded. Julia and BLAS each use one thread. Timings
come from a shared host; small differences should not be treated as stable rankings.

All solves use `maxiters = 1000` and `abstol = reltol = 1e-9`. For polyalgorithms,
the iteration budget applies to each attempted stage. The `representation` column records the requested problem/linear-solver configuration.
Matrix-free cases use `KrylovJL_LSMR()` with analytic JVP/VJP callbacks. Other cases use the default linear
solver; sparse cases supply a sparse Jacobian prototype. The automatic-Jacobian cases
omit analytic callbacks and use `AutoForwardDiff()`.

A successful return code alone is insufficient. Acceptance additionally requires:

- Every coordinate is feasible.
- Root problems: maximum absolute residual at most `1e-6`.
- Least squares: infinity norm of `u - clamp(u - J'r, lb, ub)` at most `1e-5`, and
  `sum(abs2, r)/2` no more than `reference_cost + 1e-8 * max(1, reference_cost)`.

Residuals and analytic Jacobians are recomputed outside the timed solve. Exceptions
remain explicit failed rows. Three Freudenstein–Roth starts lead all tested native
methods to a non-root local minimum; they remain failures in the reported counts.

`analyze_bounded.py` scores every ordered two-, three-, and four-stage combination of the
seven native configurations. Fallback follows solver return codes, including escalation
of `StalledSuccess`, rather than an accuracy oracle. A stage that reports success but
fails independent validation stops the sequence and counts as a failed solve. Exceptions
abort the sequence. If all stages stop without acceptance, the lowest residual infinity
norm (roots) or 2-norm (least squares) determines the returned stage. Each family has equal aggregate weight, so the diffusion size and
representation sweep does not dominate selection. Timing ratios use the fastest verified
native method for each case; coverage is reported separately from speed among successes.
The selected order is also timed directly with `FastShortcutBoundedPolyalg()`.
Algorithm objects are constructed before timing. The `bounded_default` measurements
time that explicit polyalgorithm; automatic selection and initial-guess projection in
`solve(prob)` are covered by the solver tests rather than these timings.

The paired experiment warms both leading methods before timing either, alternates their
measurement order between cases, and requests 15 samples per configuration. It checks
whether the initial timing-based ordering is stable under a second measurement protocol.

The cache experiment includes reinitialization in the timed operation and compares full
restart with `retain_best = true`. It reuses identical starts and parameters, measuring
cache reuse rather than claiming to represent an arbitrary parameter continuation.
It requests 17 samples and reports median, mean, and maximum times so that the retained
polyalgorithm's periodic first-stage reprobes are included, rather than timing only the
fast retained stage.

## Limits

These are local optimization results, not guarantees of global convergence. The suite
does not cover GPU execution, distributed problems, noisy residuals, arbitrary precision,
or every AD backend and preconditioner. Dense small problems and one sparse PDE family
cannot establish a universal ordering. The transformed baselines must use the corrected
analytic derivative chain rule; results obtained before that fix are unsuitable for
solver recommendations.

## Recorded results

The committed `results/bounded/` artifacts contain 869 measurements (79 cases × 11
configurations), 158 paired measurements, and 24 cache measurements. The bounded default
passed 76/79 cases, compared with 63/79 for the previous transformed default. Its
family-weighted geometric time ratio to the fastest accepted native method was 1.58.
Bounded trust region alone passed 75/79 at 1.41; projected Gauss–Newton passed 76/79 at
1.72. These ratios summarize different successful subsets and are not direct speedups.

The best two-stage order was bounded trust region followed by projected Gauss–Newton;
additional stages did not improve coverage. The paired experiment confirmed this order:
trust region was faster on 53/75 common successes, with a family-weighted geometric
Gauss–Newton/trust-region time ratio of 1.279. The operator subset favored Gauss–Newton
by a geometric factor of 1.593. All 79 analytic benchmark Jacobian checks passed.

In the underdetermined cache case, retaining the successful stage reduced mean time
from 13.04 ms to 1.47 ms, including periodic reprobes. The other cache cases did not show
a consistent benefit. Use the recorded mean alongside the median and maximum.

`environment.toml` identifies the host, package versions, source prerequisites, and
measurement settings. `summary.json` records coverage, timing ratios, and the ten best
orders at each sequence length. `performance.svg` plots coverage against a timing
threshold; `results.csv`, `pairwise.csv`, and `cache.csv` retain individual observations.
