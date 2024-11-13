@testitem "Modeling Toolkit Cache Indexing" tags=[:downstream] begin
    using ModelingToolkit
    using ModelingToolkit: t_nounits as t
    import NonlinearSolveBase, NonlinearSolveFirstOrder

    @parameters p d
    @variables X(t)
    eqs = [0 ~ sin(X + p) - d * sqrt(X + 1)]
    @mtkbuild nlsys = NonlinearSystem(eqs, [X], [p, d])

    # Creates an integrator.
    nlprob = NonlinearProblem(nlsys, [X => 1.0], [p => 2.0, d => 3.0])

    @testset "$integtype" for (alg, integtype) in [
        (NewtonRaphson(), NonlinearSolveFirstOrder.GeneralizedFirstOrderAlgorithmCache),
        (FastShortcutNonlinearPolyalg(), NonlinearSolveBase.NonlinearSolvePolyAlgorithmCache),
        (SimpleNewtonRaphson(), NonlinearSolveBase.NonlinearSolveNoInitCache)
    ]
        nint = init(nlprob, alg)
        @test nint isa integtype

        for (i, sym) in enumerate([X, nlsys.X, :X])
            # test both getindex and setindex!
            nint[sym] = 1.5i
            @test nint[sym] == 1.5i
        end

        for (i, sym) in enumerate([p, nlsys.p, :p])
            # test both getindex and setindex!
            nint.ps[sym] = 2.5i
            @test nint.ps[sym] == 2.5i
        end
    end
end
