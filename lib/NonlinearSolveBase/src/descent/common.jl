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
