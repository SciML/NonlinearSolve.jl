module SimpleNonlinearSolveStaticArraysExt

using SimpleNonlinearSolve

@inline SimpleNonlinearSolve.__is_extension_loaded(::Val{:StaticArrays}) = true

end
