module SimpleBracketingNonlinearSolve

using CommonSolve: CommonSolve
using SciMLBase: SciMLBase, AbstractNonlinearAlgorithm, IntervalNonlinearProblem, ReturnCode

abstract type AbstractBracketingAlgorithm <: AbstractNonlinearAlgorithm end

end
