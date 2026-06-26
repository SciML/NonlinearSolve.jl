using NonlinearSolve

# End-to-end regression for SciML/NonlinearSolve.jl#939 and
# EnzymeAD/Enzyme.jl#3130. This drives a full reverse-mode
# `Enzyme.autodiff` pass through `solve(::NonlinearProblem, NewtonRaphson())`
# on the in-place (IIP) path, exercising the `solve_up` Enzyme
# augmented_primal rule whose deeply nested `Tape{NamedTuple{...}}` type used
# to abort the worker with "GC error (probable corruption)" on macOS LTS
# (Julia 1.10) under Enzyme v0.13.150 (Enzyme.jl#3130). The probe in
# `adjoint_tests__item2.jl` only checks the `maybe_wrap_nonlinear_f`
# short-circuit; this item is the actual gradient the user cares about, and
# is restored here so CI (including macOS LTS) verifies the GC issue stays
# fixed on current Enzyme.
using SciMLSensitivity, Enzyme

function simple_loss(p)
    prob = NonlinearProblem((du, u, p) -> du[1] = u[1] - p[1] + p[2], [0.0], p)
    sol = solve(prob, NewtonRaphson())
    return sum(sol.u)
end

p = [2.0, 1.0]
dp = Enzyme.make_zero(p)
Enzyme.autodiff(
    Enzyme.Reverse, simple_loss, Enzyme.Active, Enzyme.Duplicated(p, dp)
)

@test dp ≈ [1.0, -1.0]
