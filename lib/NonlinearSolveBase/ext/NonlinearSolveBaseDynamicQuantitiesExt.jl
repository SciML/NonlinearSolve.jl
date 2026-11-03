module NonlinearSolveBaseDynamicQuantitiesExt

using NonlinearSolveBase
using DynamicQuantities: AbstractQuantity, AbstractRealQuantity, ustrip

# A DynamicQuantities type does not carry its units, so the default tolerance takes them
# from the value `u`, and the machine epsilon from its unitless numeric value
function NonlinearSolveBase.get_tolerance(
        u::Union{AbstractQuantity, AbstractRealQuantity}, ::Nothing, ::Type{T}
    ) where {T}
    v = ustrip(u)
    return T(oneunit(u) * NonlinearSolveBase.get_tolerance(v, nothing, typeof(v)))
end

end
