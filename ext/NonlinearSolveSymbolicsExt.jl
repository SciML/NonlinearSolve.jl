module NonlinearSolveSymbolicsExt

using NonlinearSolve: NonlinearSolve

NonlinearSolve.is_extension_loaded(::Val{:Symbolics}) = true

end
