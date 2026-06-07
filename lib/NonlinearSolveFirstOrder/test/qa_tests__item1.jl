using NonlinearSolveFirstOrder

using Aqua, NonlinearSolveFirstOrder

Aqua.test_all(
    NonlinearSolveFirstOrder;
    piracies = false, ambiguities = false
)
Aqua.test_piracies(NonlinearSolveFirstOrder; treat_as_own = [NonlinearLeastSquaresProblem])
Aqua.test_ambiguities(NonlinearSolveFirstOrder; recursive = false)
