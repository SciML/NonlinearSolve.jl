module NonlinearSolveBaseDynamicQuantitiesExt

using NonlinearSolveBase
using DynamicQuantities: AbstractQuantity, AbstractRealQuantity, ustrip

# A DynamicQuantities type does not carry its units, so the default tolerance takes them
# from the value `u`, and the machine epsilon from its unitless numeric value in the target
# type `T` (e.g. Float64 for a Float32 `u` in a mixed-precision bracket)
function NonlinearSolveBase.get_tolerance(
        u::Union{AbstractQuantity, AbstractRealQuantity}, ::Nothing, ::Type{T}
    ) where {T}
    v = ustrip(convert(T, u))
    return T(oneunit(u) * NonlinearSolveBase.get_tolerance(v, nothing, typeof(v)))
end

end
