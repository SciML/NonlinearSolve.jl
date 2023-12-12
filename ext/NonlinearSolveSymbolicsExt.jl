module NonlinearSolveSymbolicsExt

import NonlinearSolve, Symbolics

NonlinearSolve.is_extension_loaded(::Val{:Symbolics}) = true

end
