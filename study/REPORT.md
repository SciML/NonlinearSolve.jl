# Derivative-using bracketing methods: design study

This study starts from the `NewtonBisection` of
[NonlinearSolve.jl PR 1330](https://github.com/SciML/NonlinearSolve.jl/pull/1330) (head
`a84b9c59`). That method needs the `jac` field on `IntervalNonlinearFunction` added by
[SciMLBase.jl PR 1617](https://github.com/SciML/SciMLBase.jl/pull/1617) (head `5e42091a`).
The motivating use is locating ContinuousCallback events
([OrdinaryDiffEq.jl PR 4624](https://github.com/SciML/OrdinaryDiffEq.jl/pull/4624)).

Every number below comes from a script in this folder; the raw output is in
`study/results/`. See [Reproducing](#reproducing).

## Summary

1. **Hybrids.** Derivative methods save about 15-20% of the f evaluations of ModAB. They pay
   for that with nearly as many f' evaluations, so they only win when f' costs less than
   about 0.3 of f.
   - **Newton-accelerated ModAB is the same method as NewtonBisection.** Take Newton from the
     better end when it is accepted, else the ModAB step: the Newton step is accepted almost
     every iteration, so the fallback hardly matters, and the counts are the same as
     NewtonBisection's (389 vs 389 f-evaluations over all 18 problems).
   - **Inverse cubic Hermite (a Brent-like Newton/secant mix) gains nothing** over Newton.
   - **Halley cuts f evaluations by 29%** (64 vs 90 on the standard functions), but it needs
     f''. `IntervalNonlinearFunction` has no field for f''. With nested ForwardDiff Duals it
     costs 1.0-1.8× an f on transcendental functions, but 4.7× and 60× on two polynomials.
   - **The Newton polish after ModAB (`ModABNewton`) is the most robust design.** It uses
     the fewest f' calls, but it is only marginally cheaper than ModAB, and only when f' is
     cheap.
   - **PR 1330's NewtonBisection converges linearly at multiple roots.** It takes 88-102
     evaluations where ModAB takes 55-61.
   - **The one change clearly worth making to PR 1330 is a multiplicity-corrected Newton step
     plus lazy evaluation of `jac`.** It needs 4-14 evaluations on multiple roots, the same f
     count elsewhere, and 27% fewer `jac` calls.
2. **AD tie-in.**
   - **ForwardDiff-backed Newton is not cheaper than ModAB overall.** On the standard
     functions it costs 124-129 f-units against 108 (two runs), using each function's measured
     Dual/f cost ratio. It wins only where the Dual is nearly free (exp, cos, atan); it loses on
     polynomials and on cheap f in wall time.
   - **On an ODE interpolant the event-callback caveat holds.** DiffEqBase's in-place
     condition cannot take a Dual time (a MethodError). Through the out-of-place interpolant,
     AD is 3.7-8× slower per event than ModAB.
   - **Even a user derivative did not beat ModAB in wall time on the ODE events measured.**
   - I made AD opt-in (`autodiff = AutoForwardDiff()`), not the default.
3. **Packaging.** Keep the derivative and AD methods in BracketingNonlinearSolve.
   ADTypes and DifferentiationInterface are already dependencies of NonlinearSolveBase. Adding
   them changes neither the load time (0.534 s vs 0.525 s) nor the invalidation logs (232 vs
   232). A new sublibrary would need the kernel and the Dual-endpoint extension either
   duplicated or made public, and would gain nothing.
   - **What could go into PR 1330:** commit A (the NewtonBisection changes).
   - **What should be follow-ups:** commit C (AD, opt-in), and the Brent bug found on the way.
     I do not recommend merging `ModABNewton` (commit B).

## Branch

`interval-derivative-study`, based on the PR 1330 head. The final state passes the Core and
QA groups (see [Verification](#verification)).

| commit | content | proposed for |
|---|---|---|
| A `NewtonBisection: lazy derivative, multiplicity correction` | the Newton kernel from the end with the smaller \|f\|; `jac` only where a step starts; multiplicity-corrected step; keeps Newton points in the bracket's type; `test/rootfind_tests__item9.jl` | PR 1330 |
| B `Add ModABNewton` | ModAB's bisection phase, then the kernel of A | not recommended (kept as evidence) |
| C `NewtonBisection, ModABNewton: opt-in automatic differentiation` | `autodiff` keyword, DifferentiationInterface; ForwardDiff extension fix for Duals in the time span with no `jac` | a follow-up PR |
| D `study: ...` | this report, the scripts and the results | not for merging |

PR 1330's own tests (`rootfind_tests__item8.jl`) are unchanged on the branch. No existing test
was weakened or skipped.

## 1. Hybrids beyond rtsafe

### Candidates

I first built plain-function prototypes (`proto.jl`, run by `explore.jl`), counting
evaluations of f and f' separately:

- **`Newton`**: Newton from the end with the shorter Newton step, rtsafe acceptance (step at
  most half the step before the previous one), bisection otherwise. This is PR 1330's method
  reimplemented; its counts are within ±2 of the package's.
- **`Newton(width)`**: the same, but safeguarded by ModAB's bracket-width threshold instead
  of rtsafe. This was much worse at `abstol = 0` (513 f on all problems against 389):
  Newton converges one-sidedly, so the width does not shrink, and the threshold forces
  needless bisections.
- **`NModAB`**: the derivative-accelerated ModAB asked for. Newton from the better end when
  accepted, else the ModAB step (bisection until f looks linear, then Anderson-Björck).
  Variants accept Newton by rtsafe or by "step ≤ half the bracket".
- **`Hermite`**: the root of the inverse cubic Hermite interpolant through both ends (f and f'
  at each). Order 1+√3 when both ends move, and at least quadratic with one stale end. This
  is the derivative analog of the inverse interpolation in Brent and ModAB.
- **`Halley`**: Halley's step from the end with the shorter Newton step (needs f'').
- **`ModAB→…`**: ModAB's bisection phase until its linearity test passes, then Newton or
  Hermite (the "Newton polish").
- **Options:**
  - **`mult`**: multiply the Newton step by m = round(1/(1-ρ)) when consecutive steps from
    the same end shrink by ρ > 1/2, as at a root of multiplicity m.
  - **`lazy`**: evaluate f' only at the Newton base, chosen by smaller |f|.
  - **`push`**: move a Newton point predicted, from f''/f', to land on its base's side past
    the root by twice the predicted error.

Totals at `abstol = 0`, over the 9 standard functions of PR 1330's table and the 9 hard cases
(from `results/proto_counts.md`; "f / f'" is evaluations of f / of f'):

| prototype | standard: f / f' | hard: f / f' |
|---|---|---|
| ModAB (reference, matches the package) | 108 / 0 | 232 / 0 |
| Newton (≈ PR 1330) | 92 / 92 | 297 / 297 |
| NModAB (Newton, else ModAB step) | 92 / 92 | 297 / 297 |
| Newton + mult | 90 / 90 | 116 / 116 |
| lazy + mult | 90 / 62 | 118 / 67 |
| lazy + mult + push | 87 / 59 | 122 / 70 |
| Hermite | 89 / 89 | 301 / 301 |
| Halley (f, f', f'' per evaluation) | 64 / 64 | 248 / 248 |
| ModAB→Newton + push | 85 / 62 | 227 / 47 |
| ModAB→Hermite | 82 / 59 | 229 / 49 |
| ModAB→lazy | 92 / 49 | 226 / 31 |

What this shows:

- **NModAB is indistinguishable from Newton with a bisection fallback.** The Newton step is
  accepted in almost every iteration, so which fallback runs barely matters. The
  "step ≤ half the bracket" acceptance is worse (464 against 389 f in total): it takes
  slowly converging Newton steps that rtsafe rejects.
- **Hermite interpolation gains nothing over Newton.** One end of the bracket is usually
  stale, so it reduces to a slightly better Newton step, and the end game dominates at
  `abstol = 0`.
- **Halley needs 29% fewer evaluations**, and would be worth it only if f'' came almost free
  with f and f' (see part C of the timings below).
- **The multiplicity correction is the one large win.** Plain Newton converges linearly at
  multiple roots, and it fixes that.
- **The push gains 3% on the standard set and loses 3% on the hard set.** Not worth the
  complexity, so it is dropped.
- **Lazy evaluation of f' keeps the f count and saves 30% of the f' calls.**
- **"Newton-polish after ModAB" (`ModAB→lazy`) uses the fewest f' calls and never does worse
  than ModAB on the hard set.** It only saves 15% of the f evaluations on the standard set.

I ported two designs:

- **`NewtonBisection` (lazy + mult)** replaces the kernel of PR 1330: commit A.
- **`ModABNewton` (ModAB→lazy)**: commit B.

### Evaluation counts of the package methods

`bench_counts.jl`, full tables in `results/counts.md`. Each cell is "f/jac". For the AD
variants it is "f/Dual": calls of f with a Float argument and with a ForwardDiff Dual, which
gives f and f' together. "PR NewtonBisection" is PR 1330's method, copied verbatim into
`pr1330_newton_bisection.jl`. Brent counts are inflated about 2× by a bug (see
[Other findings](#other-findings)).

**Standard functions (PR 1330's table), `abstol = 0`:**

| problem | bracket | ModAB | ITP | Brent | PR NewtonBisection | NewtonBisection | ModABNewton | NewtonBisection AD | ModABNewton AD |
|---|---|---|---|---|---|---|---|---|---|
| u^2 - 2 | (1.0, 20.0) | 15/0 | 13/0 | 65/0 | 9/9 | 9/6 | 12/6 | 0/9 | 5/8 |
| exp(u) - 10 | (0.0, 5.0) | 12/0 | 15/0 | 37/0 | 11/11 | 8/6 | 9/5 | 0/8 | 4/6 |
| cos(u) - u | (0.0, 1.0) | 8/0 | 12/0 | 16/0 | 6/5 | 6/4 | 7/4 | 0/6 | 3/5 |
| u^3 - 2u - 5 | (2.0, 3.0) | 11/0 | 12/0 | 19/0 | 7/7 | 7/5 | 8/5 | 0/7 | 3/6 |
| log(u) - 1 | (1.0, 10.0) | 9/0 | 11/0 | 18/0 | 8/7 | 9/6 | 9/6 | 0/9 | 3/7 |
| 0.3 - (e^-u + e^-3u)/2 | (0.0, 5.0) | 12/0 | 16/0 | 22/0 | 11/10 | 13/10 | 12/5 | 0/13 | 6/7 |
| u^10 - 1 | (0.0, 1.3) | 12/0 | 19/0 | 18/0 | 9/8 | 10/8 | 12/6 | 0/10 | 6/7 |
| atan(u) - 1 | (-10.0, 20.0) | 13/0 | 15/0 | 24/0 | 11/10 | 12/8 | 10/6 | 0/12 | 4/7 |
| (u-1)^3 - 1e-3 | (0.0, 3.0) | 16/0 | 22/0 | 35/0 | 18/18 | 16/9 | 13/6 | 0/16 | 7/7 |
| **total** | | 108/0 | 135/0 | 254/0 | 90/85 | 90/62 | 92/49 | 0/90 | 41/60 |

**Hard cases, `abstol = 0`:**

| problem | bracket | ModAB | ITP | Brent | PR NewtonBisection | NewtonBisection | ModABNewton | NewtonBisection AD | ModABNewton AD |
|---|---|---|---|---|---|---|---|---|---|
| triple root (u-1)^3 | (0.0, 3.0) | 55/0 | 66/0 | 228/0 | **88/87** | 4/2 | 55/0 | 0/4 | 55/0 |
| 5-fold root (u-1)^5 | (0.0, 3.0) | 55/0 | 66/0 | 226/0 | **102/101** | 5/3 | 55/0 | 0/5 | 55/0 |
| flat u^20 - 1 | (0.0, 5.0) | 14/0 | 9/0 | 30/0 | 13/12 | 11/6 | 15/6 | 0/11 | 9/7 |
| Newton leaves: atan(u) | (-10.0, 20.0) | 12/0 | 24/0 | 22/0 | 12/11 | 12/10 | 12/9 | 0/12 | 3/10 |
| Newton leaves: tanh(10(u-0.3)) | (-1.0, 2.0) | 11/0 | 14/0 | 20/0 | 10/9 | 10/6 | 10/5 | 0/10 | 3/8 |
| f'=0 at end: u^3 - 2 | (0.0, 2.0) | 10/0 | 13/0 | 16/0 | 8/7 | 8/6 | 9/5 | 0/8 | 4/6 |
| f'=0 at end: cos(u) | (0.0, 2.0) | 11/0 | 13/0 | 51/0 | 6/6 | 6/4 | 7/4 | 0/6 | 3/5 |
| kink: max(u, 3u-1) - 0.8 | (0.0, 1.0) | 7/0 | 13/0 | **2000/0 FAIL** | 5/5 | 6/3 | 7/2 | 0/6 | 4/4 |
| jump: sign(u - 1/3) | (0.0, 1.0) | 57/0 | 56/0 | 111/0 | 56/56 | 56/27 | 56/0 | 0/56 | 56/0 |
| **total** | | 232/0 | 274/0 | 2704/0 | 300/294 | 118/67 | 226/31 | 0/118 | 192/40 |

**Totals at `abstol = 1e-8`:**

| set | ModAB | ITP | Brent | PR NewtonBisection | NewtonBisection | ModABNewton |
|---|---|---|---|---|---|---|
| standard | 104/0 | 120/0 | 186/0 | 83/83 | 82/55 | 84/43 |
| hard | 155/0 | 174/0 | 418/0 | 180/179 | 88/52 | 145/29 |

**Float32 and BigFloat, standard functions, `abstol = 0`:** every method ends with adjacent
floats or an exact zero, except Brent on `cos(u) - u` in BigFloat (MaxIters).

| type | ModAB | PR NewtonBisection | NewtonBisection | ModABNewton | NewtonBisection AD |
|---|---|---|---|---|---|
| Float32 | 95/0 | 81/75 | 79/49 | 80/37 | 0/79 |
| BigFloat | 137/0 | 109/104 | 111/81 | 110/67 | 0/111 |

**Multiple roots away from exactly representable points** (`multiple_roots.jl`,
`abstol = 0`, "f/jac"):

| f on (0, 3) | ModAB | PR NewtonBisection | NewtonBisection | ModABNewton |
|---|---|---|---|---|
| (u - 1/3)^3 | 56/0 | 87/86 | 4/2 | 56/0 |
| (u - π/4)^3 | 56/0 | 90/89 | 4/2 | 56/0 |
| (u - π/4)^5 | 56/0 | 101/100 | 4/2 | 56/0 |
| (u - π/4)^3, expanded polynomial | 22/0 | 32/31 | 4/2 | 22/0 |
| sin(u - π/4)^3 | 61/0 | 89/88 | 12/8 | 11/8 |
| (exp(u) - 2)^3 | 56/0 | 88/87 | 14/12 | 56/0 |

In every case the root was found as an exact zero of the computed f.

### Linear-convergence cases

- **Multiple roots: PR 1330's NewtonBisection is 1.5-1.9× worse than ModAB.** Newton's
  error ratio is (m-1)/m ≥ 2/3 there, which rtsafe's "half the step before the previous one"
  accepts (4/9 < 1/2). The multiplicity correction brings it to 4-14 evaluations.
- **ModABNewton does not help at multiple roots.** ModAB's linearity test never passes
  there, so it stays in bisection (sin(u - π/4)^3 is the exception).
- **Discontinuous f with f' = 0 (the jump case): every method is bisection** (56-57). The lazy
  `jac` halves the wasted derivative calls (27 against PR 1330's 56).
- **Inexact derivatives slow Newton down.** OrdinaryDiffEq PR 4624 computes f' from the ODE
  right-hand side at the interpolated state, which is not the derivative of the interpolant
  the root finder sees. On the 1000-state ODE below, this costs 8.55 f per event against
  6.67 with the interpolant's derivative. Newton's local convergence becomes linear with ratio
  \|1 - f'_interp / f'_used\|.

### Wall time

`bench_time.jl`, abstol = 0, median over BenchmarkTools samples, summed over the standard
functions; full output in `results/time.md`.

**A. The functions as they are.** f costs a few ns, so the time is mostly the solver's own
work.

| problem | ModAB | ITP | Brent | PR NewtonBisection | NewtonBisection | ModABNewton | NewtonBisection AD | ModABNewton AD |
|---|---|---|---|---|---|---|---|---|
| u^2 - 2 | 136 | 240 | 378 | 102 | 113 | 146 | 114 | 146 |
| exp(u) - 10 | 195 | 412 | 424 | 241 | 228 | 199 | 221 | 184 |
| cos(u) - u | 87 | 279 | 188 | 100 | 103 | 110 | 108 | 119 |
| u^3 - 2u - 5 | 104 | 221 | 151 | 82 | 81 | 93 | 78 | 76 |
| log(u) - 1 | 145 | 253 | 236 | 176 | 232 | 200 | 221 | 207 |
| 0.3 - (e^-u + e^-3u)/2 | 257 | 526 | 371 | 445 | 366 | 326 | 322 | 253 |
| u^10 - 1 | 158 | 442 | 254 | 198 | 227 | 277 | 245 | 287 |
| atan(u) - 1 | 333 | 560 | 476 | 275 | 300 | 263 | 301 | 293 |
| (u-1)^3 - 1e-3 | 141 | 438 | 313 | 243 | 232 | 158 | 232 | 153 |
| **total** | 1557 | 3371 | 2792 | 1861 | 1882 | 1772 | 1842 | 1718 |

**B. Expensive f** (1000 dependent multiply-adds, ~1 μs), with f' costing r times f. Timing
noise is about ±5% (ModAB does not use f', yet varies between rows).

| r | ModAB | ITP | Brent | PR NewtonBisection | NewtonBisection | ModABNewton |
|---|---|---|---|---|---|---|
| 0.0 | 116.9 | 148.4 | 267.8 | 100.4 | 99.9 | 99.8 |
| 0.1 | 115.1 | 145.1 | 267.7 | 102.0 | 100.7 | 101.6 |
| 0.25 | 115.1 | 145.1 | 267.6 | 111.9 | 108.0 | 107.3 |
| 0.5 | 115.7 | 145.1 | 267.7 | 136.2 | 125.8 | 121.3 |
| 1.0 | 115.2 | 145.1 | 267.6 | 184.7 | 161.3 | 149.3 |

**C. Cost of one ForwardDiff evaluation relative to f** (f' by a Dual, f' and f'' by a Dual of
Duals). The plain f of the polynomials is summed with SIMD, so their ratios overstate the
per-call cost. The nested-Dual numbers for u^3 - 2u - 5 (60×) and u^10 - 1 (4.7×) point to
a slow path for integer powers of nested Duals, which I did not investigate. The ratios also
move between runs: log(u) - 1 measured 0.88 in one run and 1.20 in the other.

| problem | f (ns) | f, f' (ns) | ratio | f, f', f'' (ns) | ratio |
|---|---|---|---|---|---|
| u^2 - 2 | 0.10 | 0.24 | 2.35 | 0.22 | 2.22 |
| exp(u) - 10 | 4.01 | 4.00 | 1.00 | 4.01 | 1.00 |
| cos(u) - u | 4.07 | 5.14 | 1.26 | 3.99 | 0.98 |
| u^3 - 2u - 5 | 0.16 | 0.26 | 1.65 | 9.43 | 59.74 |
| log(u) - 1 | 3.81 | 4.57 | 1.20 | 4.58 | 1.20 |
| 0.3 - (e^-u + e^-3u)/2 | 6.58 | 5.62 | 0.85 | 11.68 | 1.78 |
| u^10 - 1 | 3.43 | 4.28 | 1.25 | 15.98 | 4.66 |
| atan(u) - 1 | 5.87 | 6.03 | 1.03 | 9.08 | 1.55 |
| (u-1)^3 - 1e-3 | 0.11 | 0.24 | 2.15 | 0.24 | 2.15 |

**Reading:**

- **A, cheap f:** ModAB is the fastest in total. In this run the derivative methods are
  10-21% slower; in an earlier run PR 1330's method was within 8% of ModAB. Individual cells
  move by up to ~20% between runs at these ~100 ns times. ITP and Brent are far behind.
- **B, expensive f:**
  - At r = 0, the derivative methods are 14-15% faster than ModAB.
  - At r = 0.25 they are 3-7% faster, close to the noise.
  - At r ≥ 0.5 ModAB wins. The lazy `jac` of commit A makes NewtonBisection 8-13% faster
    than PR 1330's at r ≥ 0.5.
- **Break-even ratio r:** from the counts, ModAB's 108 against NewtonBisection's 90 + 62r
  gives r ≈ 0.29, and against ModABNewton's 92 + 49r gives r ≈ 0.33.

## 2. AD tie-in

### Implementation (commit C)

- **Keyword:** `NewtonBisection(; autodiff = nothing)` and `ModABNewton(; autodiff = nothing)`,
  with `autodiff::Union{Nothing, ADTypes.AbstractADType}`.
- **With a user `jac`:** the methods use it.
- **With `autodiff` and no `jac`:** they get f and f' in one pass through
  DifferentiationInterface. The pass uses `DI.prepare_derivative` once per solve, then
  `DI.value_and_derivative`. It is non-allocating, and type-stable for Float32, Float64
  and BigFloat (tested).
- **With neither:** the ArgumentError of PR 1330, whose error message now suggests
  `autodiff = AutoForwardDiff()`.
- **This mirrors SimpleNonlinearSolve's backend handling, minus its default.**
  SimpleNonlinearSolve uses `jac` when present, else `something(autodiff, AutoForwardDiff())`.
  I did not make ForwardDiff the default here, because of the measurements below. A default
  would silently make a derivative method slower than the default solver.
- **Why not `NonlinearSolveBase.select_forward_mode_autodiff`:** it is public, but its
  `AutoPolyesterForwardDiff` check reads `prob.u0`, which `IntervalNonlinearProblem` does
  not have. It would throw once PolyesterForwardDiff is loaded. The upstream fix is a method
  of `additional_incompatible_backend_check` for `IntervalNonlinearProblem` in
  NonlinearSolveBase.

### Standard functions: cost units

- **The comparison:** ForwardDiff-derived f+f' versus ModAB.
- **The unit:** one f evaluation; a Dual evaluation costs the measured ratio c of part C.
- **The script:** `cost_units.jl`, applied to the counts above.

| problem | Dual/f cost ratio c | ModAB (f) | NewtonBisection AD (c × Dual) | ModABNewton AD (f + c × Dual) |
|---|---|---|---|---|
| u^2 - 2 | 2.35 | 15 | 21.2 | 23.8 |
| exp(u) - 10 | 1.00 | 12 | 8.0 | 10.0 |
| cos(u) - u | 1.26 | 8 | 7.6 | 9.3 |
| u^3 - 2u - 5 | 1.65 | 11 | 11.5 | 12.9 |
| log(u) - 1 | 1.20 | 9 | 10.8 | 11.4 |
| 0.3 - (e^-u + e^-3u)/2 | 0.85 | 12 | 11.0 | 11.9 |
| u^10 - 1 | 1.25 | 12 | 12.5 | 14.8 |
| atan(u) - 1 | 1.03 | 13 | 12.4 | 11.2 |
| (u-1)^3 - 1e-3 | 2.15 | 16 | 34.4 | 22.0 |
| **total** | | 108 | 129.4 | 127.4 |

**Verdict: no, not overall.**

- **Totals:** AD-backed NewtonBisection costs 20% more than ModAB (129 against 108 units),
  and ModABNewton about the same (127). An earlier run, with slightly different measured
  ratios, gave 124 and 123.5.
- **Where AD wins:** exp, cos, atan and 0.3 - (e^-u + e^-3u)/2, where the Dual reuses the
  primal computation.
- **Where it loses:** the polynomials, where one Dual pass costs about 2× the plain f.
- **Wall time on cheap f (part A):** ModAB 1557 ns against 1842 and 1718 ns for the AD
  variants. The AD path itself is fine; the per-iteration work is simply larger than
  ModAB's.

### ODE-interpolation condition

`ode/bench_ode.jl` locates events the way DiffEqBase's `find_root` does, on steps of a real
Tsit5 integrator. The condition is g(u(t)) = |u(t)|² - r, the jump condition of quantum
trajectories, on the interpolant of one step, with a root r inside each of 40 steps and
`abstol = 0`. It compares three sources of f':

- **rhs:** ∇g·f(u(t)), with the ODE right-hand side at the interpolated state, as in
  OrdinaryDiffEq PR 4624. The state of the last condition evaluation is reused when `jac` is
  called at the same t.
- **interp:** ∇g·u'(t), from `integrator(du, t, Val{1})`, the exact derivative of the
  interpolant.
- **AD:** ForwardDiff through the out-of-place interpolant, `t ↦ g(integrator(t))`.

The ends of the step are stored, not interpolated, as in `DiffEqBase.get_condition`.

**dense 50×50 generator (rhs costs more than the interpolation), Tsit5 with abstol = reltol = 1.0e-6.** condition (interpolation + g): 90 ns; rhs mat-vec: 294 ns; interpolant derivative: 34 ns

| method | f evals per event | jac or Dual evals per event | interpolations per event | μs per event | max abs(t - t_ModAB) |
|---|---|---|---|---|---|
| ModAB | 7.28 | 0.00 | 5.22 | 0.48 | 0.0e+00 |
| PR NewtonBisection, rhs | 7.12 | 6.22 | 5.12 | 1.86 | 1.8e-15 |
| NewtonBisection, rhs | 7.30 | 4.70 | 5.33 | 1.52 | 1.8e-15 |
| ModABNewton, rhs | 8.22 | 4.62 | 6.22 | 1.69 | 1.6e-15 |
| PR NewtonBisection, interp | 6.55 | 5.65 | 4.55 | 0.70 | 1.2e-15 |
| NewtonBisection, interp | 7.12 | 3.92 | 5.12 | 0.73 | 1.6e-15 |
| ModABNewton, interp | 7.70 | 3.95 | 5.70 | 0.75 | 2.2e-15 |
| ModAB, out of place | 7.33 | 0.00 | 0.00 | 0.84 | 1.2e-15 |
| NewtonBisection, AD | 0.00 | 7.85 | 0.00 | 1.77 | 4.4e-15 |
| ModABNewton, AD | 3.00 | 7.25 | 0.00 | 2.43 | 4.4e-15 |

**tridiagonal 50×50 generator (rhs cheap), Tsit5 with abstol = reltol = 1.0e-6.** condition (interpolation + g): 88 ns; rhs mat-vec: 23 ns; interpolant derivative: 26 ns

| method | f evals per event | jac or Dual evals per event | interpolations per event | μs per event | max abs(t - t_ModAB) |
|---|---|---|---|---|---|
| ModAB | 7.78 | 0.00 | 5.72 | 0.51 | 0.0e+00 |
| PR NewtonBisection, rhs | 6.78 | 5.85 | 4.78 | 0.75 | 3.6e-15 |
| NewtonBisection, rhs | 6.85 | 4.42 | 4.85 | 0.69 | 3.6e-15 |
| ModABNewton, rhs | 7.92 | 4.40 | 5.92 | 0.60 | 3.6e-15 |
| PR NewtonBisection, interp | 6.35 | 5.47 | 4.35 | 0.69 | 4.4e-15 |
| NewtonBisection, interp | 6.55 | 3.85 | 4.55 | 0.65 | 4.4e-15 |
| ModABNewton, interp | 7.72 | 3.95 | 5.72 | 0.72 | 4.4e-15 |
| ModAB, out of place | 7.80 | 0.00 | 0.00 | 0.88 | 2.2e-15 |
| NewtonBisection, AD | 0.00 | 8.45 | 0.00 | 2.25 | 4.8e-15 |
| ModABNewton, AD | 3.00 | 6.62 | 0.00 | 2.30 | 4.0e-15 |

**tridiagonal 1000×1000 generator (rhs cheap), Tsit5 with abstol = reltol = 1.0e-6.** condition (interpolation + g): 697 ns; rhs mat-vec: 303 ns; interpolant derivative: 392 ns

| method | f evals per event | jac or Dual evals per event | interpolations per event | μs per event | max abs(t - t_ModAB) |
|---|---|---|---|---|---|
| ModAB | 8.00 | 0.00 | 5.95 | 6.97 | 0.0e+00 |
| PR NewtonBisection, rhs | 8.50 | 7.78 | 6.50 | 11.03 | 2.7e-15 |
| NewtonBisection, rhs | 8.55 | 5.95 | 6.55 | 10.14 | 2.7e-15 |
| ModABNewton, rhs | 8.93 | 5.40 | 6.92 | 9.95 | 2.7e-15 |
| PR NewtonBisection, interp | 6.45 | 5.60 | 4.45 | 8.91 | 2.7e-15 |
| NewtonBisection, interp | 6.67 | 4.12 | 4.67 | 8.13 | 2.7e-15 |
| ModABNewton, interp | 7.33 | 4.08 | 5.33 | 8.48 | 2.7e-15 |
| ModAB, out of place | 8.28 | 0.00 | 0.00 | 12.17 | 1.8e-15 |
| NewtonBisection, AD | 0.00 | 10.68 | 0.00 | 40.71 | 1.9e-14 |
| ModABNewton, AD | 3.00 | 9.78 | 0.00 | 55.58 | 1.9e-14 |

**tridiagonal 50×50 generator (rhs cheap), Tsit5 with abstol = reltol = 1.0e-8.** condition (interpolation + g): 90 ns; rhs mat-vec: 25 ns; interpolant derivative: 27 ns

| method | f evals per event | jac or Dual evals per event | interpolations per event | μs per event | max abs(t - t_ModAB) |
|---|---|---|---|---|---|
| ModAB | 7.35 | 0.00 | 5.33 | 0.47 | 0.0e+00 |
| PR NewtonBisection, rhs | 6.38 | 5.45 | 4.38 | 0.69 | 2.0e-15 |
| NewtonBisection, rhs | 6.62 | 3.92 | 4.62 | 0.67 | 1.9e-15 |
| ModABNewton, rhs | 8.03 | 3.92 | 6.03 | 0.77 | 2.0e-15 |
| PR NewtonBisection, interp | 6.67 | 5.75 | 4.67 | 0.74 | 2.2e-15 |
| NewtonBisection, interp | 7.05 | 3.88 | 5.05 | 0.72 | 2.2e-15 |
| ModABNewton, interp | 7.72 | 3.90 | 5.72 | 0.74 | 2.2e-15 |
| ModAB, out of place | 7.45 | 0.00 | 0.00 | 0.86 | 6.7e-16 |
| NewtonBisection, AD | 0.00 | 8.70 | 0.00 | 2.40 | 3.8e-15 |
| ModABNewton, AD | 3.00 | 7.20 | 0.00 | 2.51 | 3.6e-15 |

**Findings:**

- **In place, which is how DiffEqBase evaluates conditions, AD fails.** A Dual time cannot be
  written into the Float64 cache: `MethodError: no method matching Float64(::Dual{...})`
  (last line of `results/ode.md`). So the PR's caveat holds: the generic condition path
  cannot be differentiated by just passing a Dual.
- **Out of place, AD works but is 3.7-8× slower per event than in-place ModAB.** The
  interpolant allocates a Dual vector, and even ModAB out of place is 1.7-1.8× slower than
  in place.
- **With a user derivative, the derivative methods save about one condition evaluation per
  event** (ModAB 7.3-8.0 → 6.35-7.1 with the interpolant's derivative). They never beat
  ModAB in wall time here: ModAB was the fastest in all four cases, by 17-46% over the best
  derivative method.
  - **The rhs derivative is expensive when the rhs is dense:** 3.2-3.9× ModAB's time on the
    dense 50×50 generator.
  - **The interpolant derivative costs 0.3-0.56 of a condition evaluation.**
- **What would make AD viable for callbacks:** the interpolant's derivative
  (`integrator(du, t, Val{1})`, same cost as an interpolation) combined with a forward-mode
  pushforward of the user's condition in the direction (u'(t), 1). That is a DiffEqBase
  change: a Dual cache for the state (the PreallocationTools pattern) and the pushforward.
  It is not something a bracketing solver can do on its own.
- **I could not reproduce PR 4624's wall-time gain.** I do not have its quantum-trajectory
  code. Its reported condition counts (ModAB 7.9-11.3 against 5.3-7.0 per event) are larger
  than on my events, which may come from larger steps; that is not measured here.

## 3. Packaging

### Measurements

`loadtime.jl`: median of 7 fresh processes on Julia 1.12.7, `--startup-file=no`. It
compares one environment with PR 1330's head against one with this branch; both have
SciMLBase PR 1617, ForwardDiff 1.4.6 and DifferentiationInterface 0.7.21. The invalidation
column is the length of SnoopCompileCore's raw logs (`logmeths` + `logedges`), not a count
of distinct invalidated methods.

| code | pr1330 (s) | branch (s) | pr1330 invalidation log length | branch invalidation log length |
|---|---|---|---|---|
| `using BracketingNonlinearSolve` | 0.534 | 0.525 | 232 | 232 |
| `using ForwardDiff` | 0.125 | 0.132 | 438 | 438 |
| `using ForwardDiff; using BracketingNonlinearSolve` | 0.661 | 0.665 | 1527 | 1527 |
| `using NonlinearSolveBase` | 0.517 | 0.550 | 232 | 232 |
| `using SimpleNonlinearSolve` | 0.674 | 0.709 | 1376 | 1376 |

- **BracketingNonlinearSolve adds no measurable load time over NonlinearSolveBase** (both
  0.52-0.55 s in this run; an earlier run gave 0.59-0.60 s against 0.53 s).
- **Its direct dependencies:** CommonSolve, ConcreteStructs, NonlinearSolveBase,
  PrecompileTools, Reexport, SciMLBase and SciMLLogging, plus ADTypes and
  DifferentiationInterface on this branch.
- **Adding ADTypes and DifferentiationInterface costs nothing measurable.** Both are already
  in NonlinearSolveBase's `[deps]`, so the load time, and the invalidation logs with and
  without ForwardDiff, are the same within noise.
- **ForwardDiff stays a weak dependency.** It is the trigger of the existing extension for
  Dual endpoints, which the new methods also go through (tested, including Duals in both the
  time span and f, with and without `jac`).

### Options

- **(a) BracketingNonlinearSolve, recommended.**
  - **Cost:** no load-time or dependency cost, as measured above.
  - **Integration:** the methods reuse `build_bracketing_solution`, `build_exact_solution` and
    `AbstractBracketingAlgorithm`, and the ForwardDiff extension's Dual-endpoint handling
    (loops over algorithm types). Tests go in the Core group as a new
    `rootfind_tests__item9.jl`, which is the monorepo convention.
  - **Invalidations:** the new code adds methods only for its own types.
- **(b) A new sublibrary** (a "differentiable IntervalNonlinearSolve" with an AD tie-in).
  - **Cost:** the same load cost (it would depend on BracketingNonlinearSolve and
    DifferentiationInterface).
  - **Duplication:** the shared helpers and the Newton kernel would have to become public API
    of BracketingNonlinearSolve, or be duplicated. The extension for Dual endpoints would
    need its own copy, since it dispatches on the algorithm types.
  - **Overhead:** it adds a package to register and version, a `[sources]` entry, CI groups
    and docs.
  - **Benefit:** none, for about 250 lines of code.
- **(c) SimpleNonlinearSolve.**
  - **Scope:** it solves `NonlinearProblem`, not `IntervalNonlinearProblem`, and already
    depends on BracketingNonlinearSolve. Putting interval methods there would split the
    interval solvers across two packages.
  - **Load:** its load time is 0.67-0.71 s, and it hard-depends on ForwardDiff, FiniteDiff
    and LineSearch. Users of `find_root` (DiffEqBase) only need BracketingNonlinearSolve.

### How `find_root` would pick a method

DiffEqBase's `find_root` hard-codes `ModAB()` with `abstol = 0`.

- **Keep that as the default.** On the event conditions measured here ModAB was never slower
  than the derivative methods.
- **If PR 4624 goes ahead**, choose `NewtonBisection()` only when the callback provides a
  derivative. That PR should show wall-time gains on its standard callback problems
  (bouncing ball, pendulum, Lotka-Volterra), not only evaluation counts.
- **An AD path belongs in DiffEqBase** (interpolant derivative plus pushforward of the
  condition), not in `find_root`'s choice of solver.

### Proposed public API

- **`NewtonBisection(; autodiff = nothing)`.** Uses `jac` when the problem has one.
  Otherwise it uses `autodiff`, any `ADTypes.AbstractADType` that DifferentiationInterface
  supports. Without either, it throws an ArgumentError.
- **`ModABNewton`: not proposed.** It is on the branch only as evidence.
- **`IntervalNonlinearFunction(f; jac)`, as in SciMLBase PR 1617.** `jac(t, p)` returns
  df/dt at t, out of place; the bracketing methods assert out-of-place problems, so the
  in-place form `jac(J, t, p)` is unused by them.
- **Suggested for SciMLBase (not done):** a way to return f and f' together. Callbacks share
  the interpolation between the two, and a separate `jac` call either repeats it or relies on
  caching the last state, as PR 4624 does. The lazy `jac` of commit A calls `jac` at the base
  end, which is not always the last point evaluated, so such a cache misses more often: its
  interpolation counts are 0.2-0.6 per event higher than PR 1330's (4.55-5.12 against
  4.35-4.67 above), although it calls `jac` 1.5-1.9 fewer times per event.

### What goes where

- **Into PR 1330: commit A (`8460fcb5` before any rebase).** It is the same algorithm with a
  better base-end rule, lazy `jac`, the multiplicity correction, and the Float32 type fix, and
  it cherry-picks onto the PR head. Its tests are `rootfind_tests__item9.jl`; the "Evaluations"
  testset fails on the PR's code and passes with A, see [Evidence](#evidence-that-the-new-tests-discriminate).
  Behavior visible to users: the same results, fewer `jac` calls, and different f counts on
  some problems (for example 13 against 11 on `0.3 - (e^-u + e^-3u)/2`, 8 against 11 on
  `exp(u) - 10`). The PR's "Fewer evaluations than ModAB" test still passes.
- **A follow-up PR: commit C (opt-in AD).** It needs its own review of the ADTypes/DI direct
  dependencies (compat `ADTypes = "1.9"`, `DifferentiationInterface = "0.7.3"`, matching
  NonlinearSolveBase).
- **Not recommended: commit B (`ModABNewton`).** It is never much better than
  ModAB or NewtonBisection, and a second derivative method is more API to explain.
- **A separate bug-fix PR: Brent** (below).

## Other findings

- **Brent at `abstol = 0` can loop forever at adjacent floats.** On `max(u, 3u - 1) - 0.8` over
  (0, 1) it spends 2000 evaluations and returns MaxIters; the same happens on `cos(u) - u` in
  BigFloat.
  - **Cause:** at adjacent floats the inverse-quadratic step returns an endpoint, and nothing
    in `lib/BracketingNonlinearSolve/src/brent.jl` (lines 73-85) rejects a step equal to
    `left` or `right`.
  - **Second bug:** `f(c)` is re-evaluated every iteration (line 61), which doubles every
    Brent count in the tables above.
  - **History:** it is present in every registered version tested (1.0.0, 1.3.0, 1.6.0,
    1.12.0), so it is not a regression.
  - **Fix:** a three-hunk diff (guard `s == left || s == right`, and carry `fc`). It fixes the
    kink case and halves the counts (u^2 - 2: 65 → 32). A subagent ran the Core group on
    master with the fix: 18,526 tests pass.
  - **Not included on this branch:** it is unrelated to the derivative methods, and there is
    no regression test yet.
- **A `Rational` comparison cost 60 ns per solve.** My first kernel compared `ρ > 1 // 2`
  (Float64 against Rational) every iteration: 171 ns per solve on u^2 - 2, against 107 ns with
  `2ρ > 1`.
- **Allocation tests on Julia 1.10.** `@allocated` inside a testset reports 80 bytes on
  Julia 1.10 for every bracketing method, ModAB included, and 0 behind a function barrier.
  The new allocation tests use a function barrier.

## Evidence that the new tests discriminate

`discriminate.jl` runs the "Evaluations" tests of `rootfind_tests__item9.jl` against PR 1330's
method and this branch's (`results/discriminate.txt`):

```
Test Summary:                                    | Pass  Fail  Total
PR 1330 versus this branch                       |    6     4     10
  PRNewtonBisection                              |    1     4      5
    multiple roots: fewer evaluations than ModAB |          1      1
    multiple roots: fewer evaluations than ModAB |          1      1
    fewer derivatives than f                     |          1      1
    fewer derivatives than f                     |          1      1
    fewer derivatives than f                     |    1            1
  NewtonBisection                                |    5            5
```

## Verification

- **Core group, Julia 1.12.7** (`GROUP=Core Pkg.test()`, with `--check-bounds=yes`): all pass.
  - PR 1330's `NewtonBisection` testset: 2051.
  - The new "Derivative-using methods": 2860.
  - Everything else: Muller 13, Interval Nonlinear Problems 17861, Tolerance 72,
    Zero-Tolerance 56, ForwardDiff with Duals in tspan 176, ModAB non-shrinking 3,
    Flipped Signs 288, kwargs forwarding 33.
- **Core group, Julia 1.10.12:** all pass, with the same counts. This ran in a separate
  environment that `Pkg.develop`s the local SciMLBase, SciMLJacobianOperators,
  NonlinearSolveBase and BracketingNonlinearSolve, because Pkg on 1.10 ignores `[sources]`.
- **QA group, Julia 1.12.7:** 20/20 pass (Aqua, and ExplicitImports including "all qualified
  accesses are public").
  - **What QA caught:** my first extension fix used `ForwardDiff.derivative`, which is not
    public. It now uses `DifferentiationInterface.derivative` with `AutoForwardDiff()`.
- **Intermediate states:** commits A and B were checked by running
  `rootfind_tests__item8.jl` and `rootfind_tests__item9.jl` at each state (2768 tests pass
  at A; 3659 at B, with `rootfind_tests__item4.jl` included). Two later fixes were squashed
  into A and C and verified only at the final state: the function-barrier allocation test
  (A), and the public-API extension fix and opt-in AD (C).
- **Local-only configuration for all runs:** a `SciMLBase = {path = ...}` source pointing
  at the SciMLBase PR 1617 checkout, in `lib/BracketingNonlinearSolve/Project.toml` and
  `test/qa/Project.toml`. It is not committed. CI will fail until SciMLBase PR 1617 is
  released, as for PR 1330 itself.
- **Formatting:** `runic --check` passes on every changed Julia file, and `typos` passes on
  the changed folders.

## What was not verified

- **The docs build.** I edited `docs/src/solvers/bracketing_solvers.md` and
  `docs/src/native/bracketingnonlinearsolve.md` and wrote docstrings with `@ref` links, but did
  not run `docs/make.jl`.
- **Julia `pre` and the Adjoint group.** I did not run them.
- **AD backends other than ForwardDiff.** Enzyme and Mooncake were untested; only the
  unavailable-backend error was checked, with `AutoEnzyme()`.
- **Halley in the package.** Its numbers come from the prototype, with analytic f'' in the
  counts and nested Duals in the timing. I did not measure Halley with AD end to end.
- **An ITP-style projection of the Newton step.** Not tried.
- **PR 4624's claims.** Neither OrdinaryDiffEq PR 4624 nor its quantum-trajectory benchmarks
  were run; the ODE results use my own conditions on a Tsit5 integrator. The event counts
  depend on step size and tolerance; only 1e-6 and 1e-8 were measured.
- **Invalidations as distinct methods.** The invalidation numbers are raw log lengths, not
  SnoopCompile's `uinvalidated` counts; they are identical between the two branches, which is
  the comparison that matters here.
- **Timing precision.** Timings are from a 4-core cloud VM (Xeon 2.1 GHz) with ±5% noise
  between repeated runs.

## Open questions for Chris

1. Should AD be the default when a derivative method gets no `jac`, as in
   SimpleNonlinearSolve? I made it opt-in because AD-backed Newton loses to ModAB on most
   problems measured. A default would be more consistent with the rest of NonlinearSolve, but
   slower than the default solver.
2. Is a combined value-and-derivative form of `IntervalNonlinearFunction` (or of the callback
   derivative in PR 4624) worth adding to SciMLBase, so that f and f' can share an
   interpolation without the last-state cache?
3. Given that no derivative method beat ModAB in wall time on the ODE events here, what
   benchmark should PR 4624 be required to show before `find_root` switches solver?
4. Do you want the Brent fix as a separate PR (with a regression test for the kink case)?
5. Should `NonlinearSolveBase.select_forward_mode_autodiff` handle `IntervalNonlinearProblem`
   (no `u0`) so that interval methods can use it?

## Reproducing

```
# environments (Julia 1.12; julialang.org package servers not needed)
git clone https://github.com/SciML/SciMLBase.jl ../SciMLBase.jl   # next to NonlinearSolve.jl
(cd ../SciMLBase.jl && git fetch origin pull/1617/head:pr1617 && git checkout pr1617)
julia +1.12 --project=study -e 'using Pkg; Pkg.instantiate()'
julia +1.12 --project=study/ode -e 'using Pkg; Pkg.instantiate()'
bash study/run_all.sh    # writes study/results/*
```

- **`study/Project.toml` and `study/ode/Project.toml`:** use `[sources]` to point at `lib/*`
  and at `../SciMLBase.jl`.
- **Versions used:** ForwardDiff 1.4.6, DifferentiationInterface 0.7.21, ADTypes 1.24.0,
  BenchmarkTools 1.8.0, OrdinaryDiffEqTsit5 2.1.4, DiffEqBase 7.21.2.
- **Load-time comparison:** needs two environments with `BracketingNonlinearSolve`,
  `NonlinearSolveBase`, `SimpleNonlinearSolve`, `SciMLBase`, `ForwardDiff` and
  `SnoopCompileCore`. They take the three `lib/*` packages from PR 1330's head (a
  `git worktree`) and from this branch respectively, then run
  `julia study/loadtime.jl pr1330=<env1> branch=<env2>`.

| file | what |
|---|---|
| `problems.jl` | the standard functions and hard cases |
| `proto.jl`, `explore.jl` | plain-function prototypes and their counts |
| `pr1330_newton_bisection.jl` | PR 1330's NewtonBisection, verbatim, as `PRNewtonBisection` |
| `bench_counts.jl` | evaluation counts of the package methods |
| `multiple_roots.jl` | multiple roots at non-representable points |
| `bench_time.jl` | wall time (cheap f, expensive f with f' at r × f, Dual cost ratios) |
| `cost_units.jl` | AD cost in f-units from the two above |
| `ode/bench_ode.jl` | event location on Tsit5 interpolants |
| `loadtime.jl` | load time and invalidation logs |
| `discriminate.jl` | the new tests against PR 1330's method |

## Links

- NonlinearSolve.jl PR 1330 (NewtonBisection): https://github.com/SciML/NonlinearSolve.jl/pull/1330
- SciMLBase.jl PR 1617 (`jac` on IntervalNonlinearFunction): https://github.com/SciML/SciMLBase.jl/pull/1617
- OrdinaryDiffEq.jl PR 4624 (callbacks with a derivative): https://github.com/SciML/OrdinaryDiffEq.jl/pull/4624
- Brent source at master: https://github.com/SciML/NonlinearSolve.jl/blob/master/lib/BracketingNonlinearSolve/src/brent.jl
- ModAB paper (2026 improvements): https://doi.org/10.3390/a19050332
