"""
    BatchedBroyden(;
        termination_condition = NLSolveTerminationCondition(NLSolveTerminationMode.NLSolveDefault;
            abstol = nothing,
            reltol = nothing))

A low-overhead batched implementation of Broyden capable of solving multiple nonlinear
problems simultaneously.

!!! note

    To use this version, remember to load `NNlib`, i.e., `using NNlib` or
    `import NNlib` must be present in your code.
"""
struct BatchedBroyden{TC <: NLSolveTerminationCondition} <:
    AbstractBatchedNonlinearSolveAlgorithm
    termination_condition::TC
end

# Implementation of solve using Package Extensions
