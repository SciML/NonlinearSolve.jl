module SimpleNonlinearSolveDiffEqBaseExt

#using DiffEqBase: DiffEqBase

using SimpleNonlinearSolve: SimpleNonlinearSolve

SimpleNonlinearSolve.is_extension_loaded(::Val{:DiffEqBase}) = true

function SimpleNonlinearSolve.solve_adjoint_internal(args...; kwargs...)
    return DiffEqBase._solve_adjoint(args...; kwargs...)
end

end
