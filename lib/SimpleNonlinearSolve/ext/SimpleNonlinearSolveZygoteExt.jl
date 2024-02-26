module SimpleNonlinearSolveZygoteExt

import SimpleNonlinearSolve

SimpleNonlinearSolve.__is_extension_loaded(::Val{:Zygote}) = true

end