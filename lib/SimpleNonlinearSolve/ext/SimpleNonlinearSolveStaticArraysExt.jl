module SimpleNonlinearSolveStaticArraysExt

using SimpleNonlinearSolve: SimpleNonlinearSolve

@inline SimpleNonlinearSolve.__is_extension_loaded(::Val{:StaticArrays}) = true

end
