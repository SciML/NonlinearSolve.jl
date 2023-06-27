struct SimpleBatchedNewtonRaphson{AD, LS, TC <: NLSolveTerminationCondition} <:
    AbstractBatchedNonlinearSolveAlgorithm
    autodiff::AD
    linsolve::LS
    termination_condition::TC
end

# Implementation of solve using Package Extensions
