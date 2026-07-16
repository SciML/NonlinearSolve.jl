# [Nonlinear Preconditioning and Iterate Limiting (PCNR)](@id nonlinear_preconditioning)

Newton-type methods can fail or crawl on systems whose residuals are violently nonlinear
— the classic example being the exponential I-V curves of semiconductor devices. Two
classical remedies are *nonlinear preconditioning* (transform the residual so Newton sees
a tamer function) and *iterate limiting* (clip each Newton update to a physically trusted
move). NonlinearSolve.jl exposes both through two hooks on
[`NonlinearFunction`](@ref SciMLBase.NonlinearFunction):

```math
G\big(f(H(\tilde{u}, u_k), p), \tilde{u}, p\big) = 0
```

  - `precondition` — a left preconditioner `G(fu, u, p)` applied to the residual. The
    solver replaces the residual with the root-equivalent composition
    `u -> G(f(u, p), u, p)` *everywhere*: function evaluations, automatic-differentiation
    Jacobians, line-search merit functions, and termination criteria. `G` must be
    root-preserving: `G(r, u, p) = 0` if and only if `r = 0`.
  - `postcondition` — a right preconditioner / corrector `H(u_proposed, u_prev, p)`
    applied to every iterate a solver is about to accept, *before* the residual is
    evaluated or convergence is tested there (the initial guess is corrected once as
    `H(u0, u0, p)`). `H` must leave solutions fixed: `H(u, u, p) = u` at any root.

This is the solver-composition viewpoint of Brune, Knepley, Smith & Tu, *Composing
Scalable Nonlinear Algebraic Solvers* (SIAM Review 57(4), 2015): `G` and `H` may depend
on the current iterate, and the solver freezes that dependence within each iteration.
`postcondition` corresponds to PETSc SNES's `SNESLineSearchSetPostCheck` hook, and it is
exactly the corrector phase of the Predictor/Corrector Newton-Raphson (PCNR) method of
Aadithya, Keiter & Mei (Sandia) used to replace ad-hoc limiting in circuit simulators.

## Left preconditioning: taming an exponential residual

Consider solving the diode equation for the junction voltage `v` at a target current:

```math
I_s\big(e^{v/V_t} - 1\big) - I_{\text{target}} = 0
```

From an initial guess of `v = 2` volts, the residual is astronomically large
(``\sim 10^{20}``) and its slope is even larger, so every Newton step retreats by only
``\mathcal{O}(V_t) = 25\,\text{mV}`` and the solve creeps:

```@example preconditioning
using NonlinearSolve

p = (; Is = 1.0e-14, Vt = 0.025, It = 1.0e-2)
f_diode(v, p) = p.Is * expm1(v / p.Vt) - p.It

prob_plain = NonlinearProblem(f_diode, 2.0, p)
sol_plain = solve(prob_plain, NewtonRaphson())
sol_plain.retcode, sol_plain.u, sol_plain.stats.nsteps
```

The fix is a residual compression. `asinh` behaves like `log` for large arguments and
like the identity near zero, is odd and strictly monotone — so `asinh(F(v))` has exactly
the same root, but is nearly *affine* in `v` wherever the exponential dominates:

```@example preconditioning
G(fu, u, p) = asinh(fu)

prob_G = NonlinearProblem(NonlinearFunction(f_diode; precondition = G), 2.0, p)
sol_G = solve(prob_G, NewtonRaphson())
sol_G.retcode, sol_G.u, sol_G.stats.nsteps
```

Newton now converges in a handful of steps instead of dozens. Since the composition is
what the solver differentiates (via AD), the Newton step, the line-search merit function,
and the convergence test are all consistently formulated in the preconditioned residual —
note that `sol_G.resid` is the *preconditioned* residual `G(f(u))`.

If you provide `jac`, `jvp`, `vjp`, `jac_prototype`, or `sparsity` alongside
`precondition`, they must describe the derivatives/structure of the composed map, not of
the raw `f`.

!!! tip "Compress selectively"

    A compression helps precisely when it makes the *composition* closer to affine —
    `asinh ∘ exp` is nearly linear, which is why this works. Applied to a residual row
    that is already linear in `u`, the same `asinh` (or any saturating transform like
    `r/(1+|r|)`) does the opposite: the composition flattens away from the root, Newton
    steps overshoot, and the solve slows down or diverges. For systems, apply
    compression componentwise and only to the rows that are actually extreme (e.g.
    leave a linear constraint row like `vj - v` below untouched).

For `NonlinearLeastSquaresProblem`s, `precondition` re-weights the least-squares
objective to `‖G(f(u))‖²`: for consistent (zero-residual) systems the solution is
unchanged, but for genuinely overdetermined fits it changes the minimizer — which makes
the hook a natural place for residual weighting.

## Iterate limiting for a circuit model: the PCNR method

Now the classic circuit-simulation problem: a voltage source `Vs` in series with a
resistor `R` feeding a diode to ground. SPICE-family simulators solve the nodal equations
with Newton-Raphson plus *junction-voltage limiting* (`pnjlim`): a proposed update to a
diode voltage is clipped to a logarithmic move relative to the previous iterate, since a
volt-sized overshoot puts `exp(v/Vt)` far outside the region where the linearization
means anything.

The PCNR formulation makes the limited quantity an explicit unknown. Our unknowns are
`u = [v, vj]` — the node voltage and the junction voltage — tied together by a
consistency equation:

```@example preconditioning
cp = (; Vs = 5.0, R = 1.0e3, Is = 1.0e-14, Vt = 0.025)

function circuit!(r, u, p)
    v, vj = u[1], u[2]
    r[1] = (v - p.Vs) / p.R + p.Is * expm1(vj / p.Vt)  # KCL at the node
    r[2] = vj - v                                       # junction consistency
    return nothing
end

prob_plain = NonlinearProblem(NonlinearFunction(circuit!), zeros(2), cp)
sol_plain = solve(prob_plain, NewtonRaphson(); maxiters = 1000)
sol_plain.retcode, sol_plain.stats.nsteps
```

Plain Newton eventually gets there, but it takes a couple hundred iterations of
millivolt-sized creep after the first step overshoots the junction voltage to several
volts. Now add the classic SPICE3 `pnjlim` limiter as a `postcondition`. It sees the
proposed iterate and the previous accepted iterate — precisely the two pieces of
information limiting needs:

```@example preconditioning
function pnjlim(vnew, vold, vt, vcrit)
    if vnew > vcrit && abs(vnew - vold) > 2vt
        if vold > 0
            arg = 1 + (vnew - vold) / vt
            vnew = arg > 0 ? vold + vt * log(arg) : vcrit
        else
            vnew = vt * log(vnew / vt)
        end
    end
    return vnew
end

vcrit = cp.Vt * log(cp.Vt / (sqrt(2) * cp.Is))

# corrector: limit the junction voltage update, leave the node voltage alone
H!(up, uprev, p) = (up[2] = pnjlim(up[2], uprev[2], p.Vt, vcrit); nothing)

prob_lim = NonlinearProblem(NonlinearFunction(circuit!; postcondition = H!), zeros(2), cp)
sol_lim = solve(prob_lim, NewtonRaphson(); maxiters = 1000)
sol_lim.retcode, sol_lim.stats.nsteps
```

An order of magnitude fewer iterations. This *is* the PCNR method:

 1. **Predictor** — the solver's ordinary Newton step on the augmented system.
 2. **Corrector** — the `postcondition` applies the limiting functions to the proposed
    iterate.
 3. **Consistency** — the framework re-evaluates the residual and Jacobian at the
    *corrected* iterate, so the next linearization matches the state the devices were
    actually evaluated at. (Traditional device-level limiting breaks exactly this
    property, which is the inconsistency PCNR was designed to remove.)

The remaining ingredient of the PCNR paper — a Schur complement that eliminates the
augmented unknowns from the linear solve so it stays at the original MNA size — is a
linear-algebra optimization: pass a specialized solver/preconditioner via `linsolve` if
your augmented block is large.

Since the residual is evaluated at the limited iterate, the two hooks compose. Here is
the same circuit with selective `asinh` compression of the exponential KCL row on top of
the limiter:

```@example preconditioning
Gsel(fu, u, p) = (fu[1] = asinh(fu[1]); nothing)
fn = NonlinearFunction(circuit!; precondition = Gsel, postcondition = H!)
sol_both = solve(NonlinearProblem(fn, zeros(2), cp), NewtonRaphson(); maxiters = 1000)
sol_both.retcode, sol_both.stats.nsteps
```

## Constraints and projections via `postcondition`

Because `H` may enforce state exactly, the same hook covers projection-style
corrections: clamping iterates into a physical domain (positivity for concentrations,
saturations in `[0, 1]`) or pinning Dirichlet-type values so the remaining equations act
as the condensed problem. For example, protecting a `log` from a Newton overshoot into
the negative domain:

```@example preconditioning
flog(u, p) = log.(u) .- p
Hpos(up, uprev, p) = clamp.(up, 1.0e-8, Inf)

prob_pos = NonlinearProblem(NonlinearFunction(flog; postcondition = Hpos), [10.0], -2.0)
sol_pos = solve(prob_pos, NewtonRaphson())
sol_pos.retcode, sol_pos.u
```

For simple box bounds, prefer the native `lb`/`ub` support described in the
[bound constraints tutorial](bound_constraints.md); `postcondition` combined with
`lb`/`ub` is rejected since the bounds transform changes the iterate coordinates the
hook would act on.

## Semantics and caveats

  - **Signatures.** Out-of-place problems use `Gfu = G(fu, u, p)` and
    `u_new = H(u_proposed, u_prev, p)`; in-place problems mutate the first argument:
    `G(fu, u, p) -> nothing` overwrites `fu`, `H(u_proposed, u_prev, p) -> nothing`
    overwrites `u_proposed`.
  - **Where `H` acts.** `postcondition` is applied to *accepted* iterates. Line searches
    and trust regions evaluate their merit/reduction models at the unlimited trial
    points, matching PETSc post-check semantics. Limiting therefore pairs most naturally
    with plain `NewtonRaphson`; a trust region will fight a strongly active limiter
    (it converges, but slowly), and quasi-Newton secant updates can be degraded by
    aggressively clipped steps.
  - **Convergence theory.** Near a root the limiter must deactivate (`H → identity`),
    recovering Newton's local quadratic convergence; `pnjlim` and projections satisfy
    this. The Jacobian intentionally does not chain through `H` — it is a corrector
    between steps, not part of the residual.
  - **Solver support.** The first-order (`NewtonRaphson`, `TrustRegion`,
    `LevenbergMarquardt`, ...), quasi-Newton (`Broyden`, `Klement`, ...), and spectral
    (`DFSane`) families and poly-algorithms composed of them support `postcondition`;
    unsupported solvers (e.g. SimpleNonlinearSolve or external wrappers) throw an
    `ArgumentError` rather than silently ignoring it. `precondition` is a problem
    transformation and works with every solver that consumes the problem function,
    including SimpleNonlinearSolve.

## References

  - P. R. Brune, M. G. Knepley, B. F. Smith, X. Tu, *Composing Scalable Nonlinear
    Algebraic Solvers*, SIAM Review 57(4), 2015.
  - K. V. Aadithya, E. R. Keiter, T. Mei, *Predictor/Corrector Newton-Raphson (PCNR): A
    Simple, Flexible, Scalable, Modular, and Consistent Replacement for Limiting in
    Circuit Simulation*, Scientific Computing in Electrical Engineering, 2020.
  - L. W. Nagel, *SPICE2: A Computer Program to Simulate Semiconductor Circuits*, UC
    Berkeley, 1975 (the original `pnjlim`).
