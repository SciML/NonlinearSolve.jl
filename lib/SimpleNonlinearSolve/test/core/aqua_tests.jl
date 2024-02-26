@testitem "Aqua" begin
    using Aqua

    Aqua.test_all(SimpleNonlinearSolve; piracies = false, ambiguities = false)
    Aqua.test_piracies(SimpleNonlinearSolve;
        treat_as_own = [
            NonlinearProblem, NonlinearLeastSquaresProblem, IntervalNonlinearProblem])
    Aqua.test_ambiguities(SimpleNonlinearSolve; recursive = false)
end
