module SimpleNonlinearSolveDiffEqBaseExt

using DiffEqBase: DiffEqBase

using SimpleNonlinearSolve: SimpleNonlinearSolve

function SimpleNonlinearSolve.solve_adjoint_internal(args...; kwargs...)
    return DiffEqBase._solve_adjoint(args...; kwargs...)
end

end
