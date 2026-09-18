"""
    DescentResult(;
        δu = missing, u = missing, success::Bool = true, linsolve_success::Bool = true,
        extras = (;)
    )

Construct a `DescentResult` object.

### Keyword Arguments

  - `δu`: The descent direction.
  - `u`: The new iterate. This is provided only for multi-step methods currently.
  - `success`: Certain Descent Algorithms can reject a descent direction for example
    [`GeodesicAcceleration`](@ref).
  - `linsolve_success`: Whether the line search was successful.
  - `extras`: A named tuple containing intermediates computed during the solve.
    For example, [`GeodesicAcceleration`](@ref) returns `NamedTuple{(:v, :a)}` containing
    the "velocity" and "acceleration" terms.
"""
@concrete struct DescentResult
    δu
    u
    success::Bool
    linsolve_success::Bool
    extras
end

function DescentResult(;
        δu = missing, u = missing, success::Bool = true, linsolve_success::Bool = true,
        extras = (;)
    )
    @assert δu !== missing || u !== missing
    return DescentResult(δu, u, success, linsolve_success, extras)
end

"""
    TrustRegionSubproblem

Choice of trust-region subproblem solver, selected with
`TrustRegion(subproblem = ...)`:

  - `TrustRegionSubproblem.More`: solve `min ‖J δu + fu‖` subject to `‖D δu‖ ≤ Δ`
    nearly exactly via Moré's safeguarded iteration on the damping parameter
    (MINPACK `lmpar`), through [`MoreTrustRegionDescent`](@ref).
  - `TrustRegionSubproblem.Dogleg`: the classical two-piece polygonal dogleg
    approximation to the subproblem solution curve, through [`Dogleg`](@ref).

A custom [`AbstractDescentDirection`](@ref) can also be passed directly to use a
user-defined subproblem solver.
"""
EnumX.@enumx TrustRegionSubproblem begin
    Dogleg
    More
end

"""
    TrustRegionScaling

Choice of the diagonal scaling `D` in the trust-region subproblem
`min ‖J δu + fu‖` subject to `‖D δu‖ ≤ Δ`, selected with
`MoreTrustRegionDescent(scaling = ...)`:

  - `TrustRegionScaling.None`: `D = I`.
  - `TrustRegionScaling.Jacobian`: Moré's scaling `Dᵢᵢ = max(Dᵢᵢ, ‖J[:, i]‖)`,
    which never decreases across iterations and makes the trust region
    scale-covariant. Requires a concrete Jacobian.
  - `TrustRegionScaling.Auto`: engages `Jacobian` scaling only when the problem
    needs it — the decision is taken once per solve from the first Jacobian and
    a matrix-free Jacobian never engages.
"""
EnumX.@enumx TrustRegionScaling begin
    None
    Jacobian
    Auto
end

_more_scaling(s::TrustRegionScaling.T) = s
function _more_scaling(s)
    return throw(
        ArgumentError("`scaling` must be a `TrustRegionScaling` (`None`, `Jacobian`, \
                       or `Auto`), got `$(s)`.")
    )
end
