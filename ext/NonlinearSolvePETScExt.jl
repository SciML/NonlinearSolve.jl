module NonlinearSolvePETScExt

using FastClosures: @closure
using MPI: MPI
using NonlinearSolveBase: NonlinearSolveBase, get_tolerance
using NonlinearSolve: NonlinearSolve, PETScSNES
using PETSc: PETSc
using SciMLBase: SciMLBase, NonlinearProblem, ReturnCode
using SparseArrays: AbstractSparseMatrix

function SciMLBase.__solve(
        prob::NonlinearProblem, alg::PETScSNES, args...; abstol = nothing, reltol = nothing,
        maxiters = 1000, alias_u0::Bool = false, termination_condition = nothing,
        show_trace::Val{ShT} = Val(false), kwargs...) where {ShT}
    # XXX: https://petsc.org/release/manualpages/SNES/SNESSetConvergenceTest/
    termination_condition === nothing ||
        error("`PETScSNES` does not support termination conditions!")

    _f!, u0, resid = NonlinearSolve.__construct_extension_f(prob; alias_u0)
    T = eltype(prob.u0)
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

    abstol = get_tolerance(abstol, T)
    reltol = get_tolerance(reltol, T)

    nf = Ref{Int}(0)

    f! = @closure (cfx, cx, user_ctx) -> begin
        nf[] += 1
        fx = cfx isa Ptr{Nothing} ? PETSc.unsafe_localarray(T, cfx; read = false) : cfx
        x = cx isa Ptr{Nothing} ? PETSc.unsafe_localarray(T, cx; write = false) : cx
        _f!(fx, x)
        Base.finalize(fx)
        Base.finalize(x)
        return
    end

    snes = PETSc.SNES{T}(petsclib,
        alg.mpi_comm === missing ? MPI.COMM_SELF : alg.mpi_comm;
        alg.snes_options..., snes_monitor = ShT, snes_rtol = reltol,
        snes_atol = abstol, snes_max_it = maxiters)

    PETSc.setfunction!(snes, f!, PETSc.VecSeq(zero(u0)))

    if alg.autodiff === missing && prob.f.jac === nothing
        _jac! = nothing
        njac = Ref{Int}(-1)
    else
        autodiff = alg.autodiff === missing ? nothing : alg.autodiff
        _jac!, J_init = NonlinearSolve.__construct_extension_jac(
            prob, alg, u0, resid; autodiff, initial_jacobian = Val(true))

        njac = Ref{Int}(0)

        if J_init isa AbstractSparseMatrix
            PJ = PETSc.MatSeqAIJ(J_init)
            jac! = @closure (cx, J, _, user_ctx) -> begin
                njac[] += 1
                x = cx isa Ptr{Nothing} ? PETSc.unsafe_localarray(T, cx; write = false) : cx
                if J isa PETSc.AbstractMat
                    _jac!(user_ctx.jacobian, x)
                    copyto!(J, user_ctx.jacobian)
                    PETSc.assemble(J)
                else
                    _jac!(J, x)
                end
                Base.finalize(x)
                return
            end
            PETSc.setjacobian!(snes, jac!, PJ, PJ)
            snes.user_ctx = (; jacobian = J_init)
        else
            PJ = PETSc.MatSeqDense(J_init)
            jac! = @closure (cx, J, _, user_ctx) -> begin
                njac[] += 1
                x = cx isa Ptr{Nothing} ? PETSc.unsafe_localarray(T, cx; write = false) : cx
                _jac!(J, x)
                Base.finalize(x)
                J isa PETSc.AbstractMat && PETSc.assemble(J)
                return
            end
            PETSc.setjacobian!(snes, jac!, PJ, PJ)
        end
    end

    res = PETSc.solve!(u0, snes)

    _f!(resid, res)
    u_ = prob.u0 isa Number ? res[1] : res
    resid_ = prob.u0 isa Number ? resid[1] : resid

    objective = maximum(abs, resid)
    # XXX: Return Code from PETSc
    retcode = ifelse(objective ≤ abstol, ReturnCode.Success, ReturnCode.Failure)
    return SciMLBase.build_solution(prob, alg, u_, resid_; retcode, original = snes,
        stats = SciMLBase.NLStats(nf[], njac[], -1, -1, -1))
end

end
