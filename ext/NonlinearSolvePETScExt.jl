module NonlinearSolvePETScExt

using MPI: MPI
using PETSc: PETSc

using NonlinearSolveBase: NonlinearSolveBase
using NonlinearSolve: NonlinearSolve, PETScSNES
using SciMLBase: SciMLBase, NonlinearProblem, ReturnCode

using SparseArrays: AbstractSparseMatrix, SparseMatrixCSC, sparse, rowvals, nonzeros, nzrange

# Helper function to copy a Julia matrix to a PETSc matrix
function copy_to_petsc_mat!(J_petsc, J_julia::AbstractMatrix)
    m, n = size(J_julia)
    for j in 1:n
        for i in 1:m
            J_petsc[i, j] = J_julia[i, j]
        end
    end
    return nothing
end

# For sparse matrices, iterate over non-zero values
function copy_to_petsc_mat!(J_petsc, J_julia::SparseMatrixCSC)
    rows = rowvals(J_julia)
    vals = nonzeros(J_julia)
    m, n = size(J_julia)
    for j in 1:n
        for ii in nzrange(J_julia, j)
            i = rows[ii]
            J_petsc[i, j] = vals[ii]
        end
    end
    return nothing
end

function SciMLBase.__solve(
        prob::NonlinearProblem, alg::PETScSNES, args...;
        abstol = nothing, reltol = nothing,
        maxiters = 1000, alias = SciMLBase.NonlinearAliasSpecifier(alias_u0 = false), termination_condition = nothing,
        show_trace::Val = Val(false), kwargs...
    )
    if haskey(kwargs, :alias_u0)
        alias = SciMLBase.NonlinearAliasSpecifier(alias_u0 = kwargs[:alias_u0])
    end
    alias_u0 = alias.alias_u0
    # XXX: https://petsc.org/release/manualpages/SNES/SNESSetConvergenceTest/
    NonlinearSolveBase.assert_extension_supported_termination_condition(
        termination_condition, alg; abs_norm_supported = false
    )

    f_wrapped!, u0,
        resid = NonlinearSolveBase.construct_extension_function_wrapper(
        prob; alias_u0
    )
    T = eltype(u0)

    if alg.petsclib === missing
        petsclibidx = findfirst(PETSc.petsclibs) do petsclib
            petsclib isa PETSc.PetscLibType{T}
        end

        if petsclibidx === nothing
            error(lazy"No compatible PETSc library found for element type $(T). Pass in a \
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

    # PETSc 0.4 callback signature: f!(fx, snes, x) returning PetscInt(0) for success
    # PETSc 0.4.2 passes PetscVec objects to callbacks
    function f!(cfx, snes_arg, cx)
        nf[] += 1
        # Use withlocalarray! with multiple vectors to get Julia array views
        # cx: read-only, cfx: write-only
        PETSc.withlocalarray!(cx, cfx; read = (true, false), write = (false, true)) do x_arr, fx_arr
            f_wrapped!(fx_arr, x_arr)
        end
        return PETSc.LibPETSc.PetscInt(0)
    end

    snes = PETSc.SNES(
        petsclib,
        alg.mpi_comm === missing ? MPI.COMM_SELF : alg.mpi_comm;
        alg.snes_options..., snes_monitor = show_trace isa Val{true}, snes_rtol = reltol,
        snes_atol = abstol, snes_max_it = maxiters
    )

    PETSc.setfunction!(snes, f!, PETSc.VecSeq(petsclib, MPI.COMM_SELF, zero(u0)))

    njac = Ref{Int}(-1)
    if alg.autodiff !== missing || prob.f.jac !== nothing
        autodiff = alg.autodiff === missing ? nothing : alg.autodiff
        if prob.u0 isa Number
            jac! = NonlinearSolveBase.construct_extension_jac(
                prob, alg, prob.u0, prob.u0; autodiff
            )
            J_init = zeros(T, 1, 1)
        else
            jac!,
                J_init = NonlinearSolveBase.construct_extension_jac(
                prob, alg, u0, resid; autodiff, initial_jacobian = Val(true)
            )
        end

        njac = Ref{Int}(0)

        mpi_comm = alg.mpi_comm === missing ? MPI.COMM_SELF : alg.mpi_comm
        if J_init isa AbstractSparseMatrix
            PJ = PETSc.MatCreateSeqAIJ(petsclib, mpi_comm, J_init)
        else
            PJ = PETSc.MatSeqDense(petsclib, J_init)
        end

        # PETSc 0.4 callback signature: jac!(J, snes, x) returning PetscInt(0) for success
        # Use J_init as intermediate storage for the Julia Jacobian
        function jac_fn!(J, snes_arg, cx)
            njac[] += 1
            PETSc.withlocalarray!(cx; read = true, write = false) do x_arr
                jac!(snes_arg.user_ctx.jacobian, x_arr)
                copy_to_petsc_mat!(J, snes_arg.user_ctx.jacobian)
                PETSc.assemble!(J)
            end
            return PETSc.LibPETSc.PetscInt(0)
        end
        PETSc.setjacobian!(snes, jac_fn!, PJ, PJ)
        snes.user_ctx = (; jacobian = J_init)
    end

    # Create PETSc vector for solution and solve
    x_petsc = PETSc.VecSeq(petsclib, MPI.COMM_SELF, copy(u0))
    PETSc.solve!(x_petsc, snes)

    # Copy solution back to Julia array
    res = similar(u0)
    PETSc.withlocalarray!(x_petsc; read = true) do x_arr
        copyto!(res, x_arr)
    end

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
