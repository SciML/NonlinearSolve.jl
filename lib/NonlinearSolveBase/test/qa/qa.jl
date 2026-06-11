using Aqua, NonlinearSolveBase
using NonlinearSolveBase: AbstractNonlinearProblem, NonlinearProblem, SciMLBase

Aqua.test_all(
    NonlinearSolveBase; piracies = false, ambiguities = false, stale_deps = false
)
Aqua.test_stale_deps(NonlinearSolveBase; ignore = [:TimerOutputs])
Aqua.test_piracies(
    NonlinearSolveBase,
    treat_as_own = [AbstractNonlinearProblem, NonlinearProblem, SciMLBase.HomotopyProblem]
)
Aqua.test_ambiguities(NonlinearSolveBase; recursive = false)
