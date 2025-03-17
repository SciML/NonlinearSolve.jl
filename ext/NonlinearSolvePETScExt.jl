module NonlinearSolvePETScExt

using FastClosures: @closure

using MPI: MPI
using PETSc: PETSc

using NonlinearSolveBase: NonlinearSolveBase
using NonlinearSolve: NonlinearSolve, PETScSNES
using SciMLBase: SciMLBase, NonlinearProblem, ReturnCode

using SparseArrays: AbstractSparseMatrix

function SciMLBase.__solve(
        prob::NonlinearProblem, alg::PETScSNES, args...;
        abstol = nothing, reltol = nothing,
        maxiters = 1000, alias_u0::Bool = false, termination_condition = nothing,
        show_trace::Val = Val(false), kwargs...
)
    if !MPI.Initialized()
        @warn "MPI not initialized. Initializing MPI with MPI.Init()." maxlog = 1
        MPI.Init()
    end

    # XXX: https://petsc.org/release/manualpages/SNES/SNESSetConvergenceTest/
    NonlinearSolveBase.assert_extension_supported_termination_condition(
        termination_condition, alg; abs_norm_supported = false
    )

    f_wrapped!, u0, resid = NonlinearSolveBase.construct_extension_function_wrapper(
        prob; alias_u0
    )
    T = eltype(u0)
    @assert T ∈ PETSc.scalar_types

    if alg.petsclib === missing
        petsclibidx = findfirst(PETSc.petsclibs) do petsclib
            petsclib isa PETSc.PetscLibType{T}
        end

        if petsclibidx === nothing
            error("No compatible PETSc library found for element type $(T). Pass in a \
                   custom `petsclib` via `PETScSNES(; petsclib = <petsclib>, ....)`.")
        end
        petsclib = PETSc.petsclibs[petsclibidx]
    else
        petsclib = alg.petsclib
    end
    PETSc.initialized(petsclib) || PETSc.initialize(petsclib)

    abstol = NonlinearSolveBase.get_tolerance(abstol, T)
    reltol = NonlinearSolveBase.get_tolerance(reltol, T)

    nf = Ref{Int}(0)

    f! = @closure (cfx, cx, user_ctx) -> begin
        nf[] += 1
        fx = cfx isa Ptr{Nothing} ? PETSc.unsafe_localarray(T, cfx; read = false) : cfx
        x = cx isa Ptr{Nothing} ? PETSc.unsafe_localarray(T, cx; write = false) : cx
        f_wrapped!(fx, x)
        Base.finalize(fx)
        Base.finalize(x)
        return
    end

    snes = PETSc.SNES{T}(
        petsclib,
        alg.mpi_comm === missing ? MPI.COMM_SELF : alg.mpi_comm;
        alg.snes_options..., snes_monitor = show_trace isa Val{true}, snes_rtol = reltol,
        snes_atol = abstol, snes_max_it = maxiters
    )

    PETSc.setfunction!(snes, f!, PETSc.VecSeq(zero(u0)))

    njac = Ref{Int}(-1)
    # `missing` -> let PETSc compute the Jacobian using finite differences
    if alg.autodiff !== missing
        autodiff = alg.autodiff === missing ? nothing : alg.autodiff

        if prob.u0 isa Number
            jac! = NonlinearSolveBase.construct_extension_jac(
                prob, alg, prob.u0, prob.u0; autodiff
            )
            J_init = zeros(T, 1, 1)
        else
            jac!, J_init = NonlinearSolveBase.construct_extension_jac(
                prob, alg, u0, resid; autodiff, initial_jacobian = Val(true)
            )
        end

        njac = Ref{Int}(0)

        if J_init isa AbstractSparseMatrix
            PJ = PETSc.MatSeqAIJ(J_init)
            jac_fn! = @closure (cx, J, _, user_ctx) -> begin
                njac[] += 1
                x = cx isa Ptr{Nothing} ? PETSc.unsafe_localarray(T, cx; write = false) : cx
                if J isa PETSc.AbstractMat
                    jac!(user_ctx.jacobian, x)
                    copyto!(J, user_ctx.jacobian)
                    PETSc.assemble(J)
                else
                    jac!(J, x)
                end
                Base.finalize(x)
                return
            end
            PETSc.setjacobian!(snes, jac_fn!, PJ, PJ)
            snes.user_ctx = (; jacobian = J_init)
        else
            PJ = PETSc.MatSeqDense(J_init)
            jac_fn! = @closure (cx, J, _, user_ctx) -> begin
                njac[] += 1
                x = cx isa Ptr{Nothing} ? PETSc.unsafe_localarray(T, cx; write = false) : cx
                jac!(J, x)
                Base.finalize(x)
                J isa PETSc.AbstractMat && PETSc.assemble(J)
                return
            end
            PETSc.setjacobian!(snes, jac_fn!, PJ, PJ)
        end
    end

    res = PETSc.solve!(u0, snes)

    f_wrapped!(resid, res)
    u_res = prob.u0 isa Number ? res[1] : res
    resid_res = prob.u0 isa Number ? resid[1] : resid

    objective = maximum(abs, resid)
    # XXX: Return Code from PETSc
    retcode = ifelse(objective ≤ abstol, ReturnCode.Success, ReturnCode.Failure)
    return SciMLBase.build_solution(
        prob, alg, u_res, resid_res;
        retcode, original = snes,
        stats = SciMLBase.NLStats(nf[], njac[], -1, -1, -1)
    )
end

end
