struct BatchedBroyden{TC <: NLSolveTerminationCondition} <:
       AbstractBatchedNonlinearSolveAlgorithm
    termination_condition::TC
end

# Implementation of solve using Package Extensions
