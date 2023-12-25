module SimpleNonlinearSolvePolyesterForwardDiffExt

using SimpleNonlinearSolve, PolyesterForwardDiff

@inline SimpleNonlinearSolve.__is_extension_loaded(::Val{:PolyesterForwardDiff}) = true

@inline function SimpleNonlinearSolve.__polyester_forwarddiff_jacobian!(f!::F, y, J, x,
        chunksize) where {F}
    PolyesterForwardDiff.threaded_jacobian!(f!, y, J, x, chunksize)
    return J
end

end