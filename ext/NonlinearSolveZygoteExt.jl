module NonlinearSolveZygoteExt

using NonlinearSolve: NonlinearSolve

NonlinearSolve.is_extension_loaded(::Val{:Zygote}) = true

end
