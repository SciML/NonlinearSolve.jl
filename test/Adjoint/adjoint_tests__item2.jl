using NonlinearSolve

# Regression for SciML/NonlinearSolve.jl#939: `maybe_wrap_nonlinear_f` used
# to unconditionally wrap AutoSpecialize IIP array problems in a
# `FunctionWrappersWrapper` whose `ptrtoint`/`store` patterns defeat
# Enzyme's static activity analysis. The fix short-circuits the wrap on the
# outer-AD path via `EnzymeCore.within_autodiff()`. This probes that
# short-circuit directly: outside Enzyme the problem wraps; inside an
# `Enzyme.autodiff` pass it returns `prob.f.f` unchanged.
#
# End-to-end gradient correctness (the full
# `Enzyme.autodiff(Reverse, simple_loss, Active, Duplicated(p, dp))`
# over `solve(::NonlinearProblem)`) is not verified here because compiling
# the `solve_up` Enzyme augmented_primal rule for an IIP problem triggers
# a "GC error (probable corruption)" abort on macOS LTS in Enzyme v0.13.150
# — a separate Enzyme/Julia 1.10 GC issue, not a NonlinearSolveBase bug.
# Has been manually verified to return `dp ≈ [1.0, -1.0]` on Julia 1.11.9
# and 1.12.6 with the patch + SciMLSensitivity loaded.
using NonlinearSolveBase, SciMLBase, Enzyme

resid!(du, u, p) = (du .= u .- p; nothing)
f = NonlinearFunction{true, SciMLBase.AutoSpecialize}(
    resid!, resid_prototype = zeros(2)
)
prob = NonlinearProblem(f, [1.0, 2.0], [0.5, 0.25])

# Outside Enzyme: this problem normally wraps.
@test NonlinearSolveBase.is_fw_wrapped(
    NonlinearSolveBase.maybe_wrap_nonlinear_f(prob)
)

# Inside Enzyme.autodiff: wrapping is skipped, `maybe_wrap_nonlinear_f`
# returns the raw function. The probe returns 0.0 in the unwrapped branch
# and `sum(p)` (nonzero) in the wrapped branch, so a return of 0.0 with
# all-zero gradient confirms the short-circuit fired.
probe = let prob = prob
    p -> begin
        wrapped = NonlinearSolveBase.maybe_wrap_nonlinear_f(prob)
        NonlinearSolveBase.is_fw_wrapped(wrapped) ? sum(p) : zero(eltype(p))
    end
end

p_test = [0.5, 0.25]
dp = Enzyme.make_zero(p_test)
_, primal = Enzyme.autodiff(
    Enzyme.ReverseWithPrimal, probe, Enzyme.Active,
    Enzyme.Duplicated(p_test, dp)
)
@test primal == 0.0
@test dp == zeros(2)
