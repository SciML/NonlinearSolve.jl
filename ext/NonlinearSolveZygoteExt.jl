module NonlinearSolveZygoteExt

import NonlinearSolve, Zygote

NonlinearSolve.is_extension_loaded(::Val{:Zygote}) = true

end
