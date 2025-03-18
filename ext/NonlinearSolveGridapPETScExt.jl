module NonlinearSolveGridapPETScExt

using Gridap: Gridap, Algebra
using GridapPETSc: GridapPETSc

using NonlinearSolveBase: NonlinearSolveBase
using NonlinearSolve: NonlinearSolve, GridapPETScSNES
using SciMLBase: SciMLBase, NonlinearProblem, ReturnCode

using ConcreteStructs: @concrete
using FastClosures: @closure

@concrete struct NonlinearSolveOperator <: Algebra.NonlinearOperator
    f!
    jac!
    initial_guess_cache
    resid_prototype
    jacobian_prototype
end

function Algebra.residual!(b::AbstractVector, op::NonlinearSolveOperator, x::AbstractVector)
    op.f!(b, x)
end

function Algebra.jacobian!(
        A::AbstractMatrix, op::NonlinearSolveOperator, x::AbstractVector
)
    op.jac!(A, x)
end

function Algebra.zero_initial_guess(op::NonlinearSolveOperator)
    fill!(op.initial_guess_cache, 0)
    return op.initial_guess_cache
end

function Algebra.allocate_residual(op::NonlinearSolveOperator, ::AbstractVector)
    fill!(op.resid_prototype, 0)
    return op.resid_prototype
end

function Algebra.allocate_jacobian(op::NonlinearSolveOperator, ::AbstractVector)
    fill!(op.jacobian_prototype, 0)
    return op.jacobian_prototype
end

# TODO: Later we should just wrap `Gridap` generally and pass in `PETSc` as the solver
function SciMLBase.__solve(
        prob::NonlinearProblem, alg::GridapPETScSNES, args...;
        abstol = nothing, reltol = nothing,
        maxiters = 1000, alias_u0::Bool = false, termination_condition = nothing,
        show_trace::Val = Val(false), kwargs...
)
    # XXX: https://petsc.org/release/manualpages/SNES/SNESSetConvergenceTest/
    NonlinearSolveBase.assert_extension_supported_termination_condition(
        termination_condition, alg; abs_norm_supported = false
    )

    f_wrapped!, u0, resid = NonlinearSolveBase.construct_extension_function_wrapper(
        prob; alias_u0
    )
    T = eltype(u0)

    abstol = NonlinearSolveBase.get_tolerance(abstol, T)
    reltol = NonlinearSolveBase.get_tolerance(reltol, T)

    nf = Ref{Int}(0)

    f! = @closure (fx, x) -> begin
        nf[] += 1
        f_wrapped!(fx, x)
        return fx
    end

    if prob.u0 isa Number
        jac! = NonlinearSolveBase.construct_extension_jac(
            prob, alg, prob.u0, prob.u0; alg.autodiff
        )
        J_init = zeros(T, 1, 1)
    else
        jac!, J_init = NonlinearSolveBase.construct_extension_jac(
            prob, alg, u0, resid; alg.autodiff, initial_jacobian = Val(true)
        )
    end

    njac = Ref{Int}(-1)
    jac_fn! = @closure (J, x) -> begin
        njac[] += 1
        jac!(J, x)
        return J
    end

    nop = NonlinearSolveOperator(f!, jac_fn!, u0, resid, J_init)

    petsc_args = [
        "-snes_rtol", string(reltol), "-snes_atol", string(abstol),
        "-snes_max_it", string(maxiters)
    ]
    for (k, v) in pairs(alg.snes_options)
        push!(petsc_args, "-$(k)")
        push!(petsc_args, string(v))
    end
    show_trace isa Val{true} && push!(petsc_args, "-snes_monitor")

    # TODO: We can reuse the cache returned from this function
    sol_u = GridapPETSc.with(args = petsc_args) do
        sol_u = copy(u0)
        Algebra.solve!(sol_u, GridapPETSc.PETScNonlinearSolver(), nop)
        return sol_u
    end

    f_wrapped!(resid, sol_u)
    u_res = prob.u0 isa Number ? sol_u[1] : sol_u
    resid_res = prob.u0 isa Number ? resid[1] : resid

    objective = maximum(abs, resid)
    retcode = ifelse(objective ≤ abstol, ReturnCode.Success, ReturnCode.Failure)
    return SciMLBase.build_solution(
        prob, alg, u_res, resid_res;
        retcode, stats = SciMLBase.NLStats(nf[], njac[], -1, -1, -1)
    )
end

end