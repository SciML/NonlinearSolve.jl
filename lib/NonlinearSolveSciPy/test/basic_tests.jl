@testitem "Package loads and algorithms defined" begin
    using Test, NonlinearSolveSciPy
    @test isdefined(NonlinearSolveSciPy, :SciPyLeastSquares)
    @test isdefined(NonlinearSolveSciPy, :SciPyRoot)
end
