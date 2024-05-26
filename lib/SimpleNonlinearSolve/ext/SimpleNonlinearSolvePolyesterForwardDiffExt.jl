module SimpleNonlinearSolvePolyesterForwardDiffExt

using PolyesterForwardDiff: PolyesterForwardDiff
using SimpleNonlinearSolve: SimpleNonlinearSolve

@inline SimpleNonlinearSolve.__is_extension_loaded(::Val{:PolyesterForwardDiff}) = true

@inline function SimpleNonlinearSolve.__polyester_forwarddiff_jacobian!(
        f!::F, y, J, x, chunksize) where {F}
    PolyesterForwardDiff.threaded_jacobian!(f!, y, J, x, chunksize)
    return J
end

@inline function SimpleNonlinearSolve.__polyester_forwarddiff_jacobian!(
        f::F, J, x, chunksize) where {F}
    PolyesterForwardDiff.threaded_jacobian!(f, J, x, chunksize)
    return J
end

end
