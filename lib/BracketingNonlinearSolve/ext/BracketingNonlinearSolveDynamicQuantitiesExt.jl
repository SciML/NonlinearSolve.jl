module BracketingNonlinearSolveDynamicQuantitiesExt

using BracketingNonlinearSolve: BracketingNonlinearSolve
using DynamicQuantities: UnionAbstractQuantity, dimension, ustrip

# The type of a DynamicQuantities value does not carry its dimension, so promotion cannot
# tell whether a tolerance and a residual compare. The residual tolerance takes the numeric
# value of `abstol` (stored in SI units, so a commensurate tolerance keeps its physical
# size) in the dimension of the residual (`oneunit` keeps the dimension and the quantity
# type of `fx`). The value keeps the precision of `abstol`, which may differ from that of
# the residual.
function BracketingNonlinearSolve.Impl.residual_tolerance(
        abstol::UnionAbstractQuantity, fx::UnionAbstractQuantity
    )
    tol = ustrip(abstol) * oneunit(abs(fx))
    @assert dimension(tol) == dimension(fx)
    return tol
end

# A plain residual: the stripped numeric tolerance
function BracketingNonlinearSolve.Impl.residual_tolerance(abstol::UnionAbstractQuantity, fx)
    return ustrip(abstol)
end

# A plain tolerance with a residual with units: the tolerance in the units of the residual
function BracketingNonlinearSolve.Impl.residual_tolerance(abstol, fx::UnionAbstractQuantity)
    return abstol * oneunit(abs(fx))
end

end
