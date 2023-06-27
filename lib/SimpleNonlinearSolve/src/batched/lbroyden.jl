struct BatchedLBroyden{TC <: NLSolveTerminationCondition} <:
    AbstractBatchedNonlinearSolveAlgorithm
    termination_condition::TC
    threshold::Int
end

# Implementation of solve using Package Extensions