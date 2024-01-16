module NonlinearSolvePolyesterForwardDiffExt

using NonlinearSolve, PolyesterForwardDiff

NonlinearSolve.is_extension_loaded(::Val{:PolyesterForwardDiff}) = true

end
