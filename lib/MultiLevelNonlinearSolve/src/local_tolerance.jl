"""
    LocalToleranceSchedule(; schedule = :quadratic, C = 1.0, ceil = 1.0e-4,
                             tol_init = ceil, floor_rel = 1.0e-2)

How accurately the local ensemble is solved at each global iteration — the *local forcing*
knob. This is the nonlinear counterpart of the linear forcing an `EisenstatWalkerForcing`
applies to `S·δū = -R̄`; the two are independent and can be used together.

Solving the local problems to full precision at every global iteration is wasteful while the
global iterate is still far from the root. Perturbed-Newton theory bounds the global error by
`e_{k+1} ≲ C₁e_k² + C₂L_q·ε_k + C₃ε_k·e_k` for a local accuracy `ε_k`, so the global rate
survives as long as `ε_k` shrinks at least as fast as the rate being claimed:

| `schedule`   | `ε_k`             | pair with              |
|:-------------|:------------------|:-----------------------|
| `:quadratic` | `C·‖R̄_k‖²`        | `jacobian_reuse = :always` |
| `:linear`    | `C·‖R̄_k‖`         | `jacobian_reuse = :chord`  |
| `:fixed`     | `tol_init`        | either                     |

The tolerance is clamped to `[floor_rel * abstol, ceil]`. The floor bounds how accurate the
committed internal variables can ever be, so the residual of the *unreduced* problem at the
returned root is only good to about `abstol + L_q·floor` — set `local_tolerance = nothing`
(the default) or a `:fixed` schedule when you need the tight bound.

### Fields

  - `schedule`: `:quadratic`, `:linear` or `:fixed`.
  - `C`: scale factor, absorbing the local Lipschitz constant and units.
  - `ceil`: the loosest tolerance ever used.
  - `tol_init`: the tolerance used before any residual exists.
  - `floor_rel`: the floor, relative to the solve's `abstol`, resolved at `init`.
"""
struct LocalToleranceSchedule{T}
    schedule::Symbol
    C::T
    ceil::T
    tol_init::T
    floor_rel::T
end

function LocalToleranceSchedule(;
        schedule::Symbol = :quadratic, C = 1.0, ceil = 1.0e-4, tol_init = ceil,
        floor_rel = 1.0e-2
    )
    schedule in (:quadratic, :linear, :fixed) || throw(
        ArgumentError(
            "`schedule` must be one of `:quadratic`, `:linear` or `:fixed`; got \
             `$(Meta.quot(schedule))`."
        )
    )
    C, ceil, tol_init, floor_rel = promote(C, ceil, tol_init, floor_rel)
    return LocalToleranceSchedule(schedule, C, ceil, tol_init, floor_rel)
end

"Exponent of `‖R̄‖` in the schedule; `0` marks a schedule that never updates."
local_forcing_exponent(s::LocalToleranceSchedule) =
    s.schedule === :quadratic ? 2 : s.schedule === :linear ? 1 : 0

"""
    LocalForcingParameters(p, tol)

The parameter object a [`MultiLevelNewton`](@ref) cache hands to the condensed residual, its
Schur assembler and `commit_internal!` when a [`LocalToleranceSchedule`](@ref) is configured.
It carries the user's own parameters plus the tolerance cell for the current global
iteration; read them with [`user_parameters`](@ref) and [`local_tolerance`](@ref), both of
which also work on a bare `p`, so the same callbacks run unchanged with and without a
schedule.

The cell belongs to one cache. Two concurrent solves of the same problem therefore get
independent tolerances, which is what makes a trial residual reproducible.
"""
struct LocalForcingParameters{P, T}
    p::P
    tol::Base.RefValue{T}
end

"""
    local_tolerance(p)

The local-solve tolerance for the global iteration in progress, or `nothing` when no
[`LocalToleranceSchedule`](@ref) is configured. Read it once on the host and pass the value
into the local solves; do not hold on to the parameter object inside a kernel.
"""
local_tolerance(::Any) = nothing
local_tolerance(p::LocalForcingParameters) = p.tol[]

"""
    user_parameters(p)

The parameters the problem was built with, unwrapping the [`LocalForcingParameters`](@ref)
a local-forcing schedule adds. The identity on any other parameter object.
"""
user_parameters(p::Any) = p
user_parameters(p::LocalForcingParameters) = p.p
