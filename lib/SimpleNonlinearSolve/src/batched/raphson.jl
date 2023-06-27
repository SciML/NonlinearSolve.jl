"""
    SimpleBatchedNewtonRaphson(; chunk_size = Val{0}(),
        autodiff = Val{true}(),
        diff_type = Val{:forward},
        termination_condition = NLSolveTerminationCondition(NLSolveTerminationMode.NLSolveDefault;
            abstol = nothing,
            reltol = nothing))

A low-overhead batched implementation of Newton-Raphson capable of solving multiple
nonlinear problems simultaneously.

!!! note

    To use the `batched` version, remember to load `AbstractDifferentiation` and
    `LinearSolve`.
"""
struct SimpleBatchedNewtonRaphson{AD, LS, TC <: NLSolveTerminationCondition} <:
    AbstractBatchedNonlinearSolveAlgorithm
    autodiff::AD
    linsolve::LS
    termination_condition::TC
end

# Implementation of solve using Package Extensions
