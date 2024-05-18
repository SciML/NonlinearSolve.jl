@testitem "Modeling Toolkit Cache Indexing" tags=[:downstream] begin
    using ModelingToolkit
    using ModelingToolkit: t_nounits as t

    @parameters p d
    @variables X(t)
    eqs = [0 ~ sin(X + p) - d * sqrt(X + 1)]
    @mtkbuild nlsys = NonlinearSystem(eqs, [X], [p, d])

    # Creates an integrator.
    nlprob = NonlinearProblem(nlsys, [X => 1.0], [p => 2.0, d => 3.0])

    @testset "GeneralizedFirstOrderAlgorithmCache" begin
        nint = init(nlprob, NewtonRaphson())
        @test nint isa NonlinearSolve.GeneralizedFirstOrderAlgorithmCache

        @test nint[X] == 1.0
        @test nint[nlsys.X] == 1.0
        @test nint[:X] == 1.0
        @test nint.ps[p] == 2.0
        @test nint.ps[nlsys.p] == 2.0
        @test nint.ps[:p] == 2.0
    end

    @testset "NonlinearSolvePolyAlgorithmCache" begin
        nint = init(nlprob, FastShortcutNonlinearPolyalg())
        @test nint isa NonlinearSolve.NonlinearSolvePolyAlgorithmCache

        @test nint[X] == 1.0
        @test nint[nlsys.X] == 1.0
        @test nint[:X] == 1.0
        @test nint.ps[p] == 2.0
        @test nint.ps[nlsys.p] == 2.0
        @test nint.ps[:p] == 2.0
    end

    @testset "NonlinearSolveNoInitCache" begin
        nint = init(nlprob, SimpleNewtonRaphson())
        @test nint isa NonlinearSolve.NonlinearSolveNoInitCache

        @test nint[X] == 1.0
        @test nint[nlsys.X] == 1.0
        @test nint[:X] == 1.0
        @test nint.ps[p] == 2.0
        @test nint.ps[nlsys.p] == 2.0
        @test nint.ps[:p] == 2.0
    end
end
