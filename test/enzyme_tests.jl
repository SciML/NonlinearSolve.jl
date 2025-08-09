@testitem "PolyAlgorithms with Enzyme" tags=[:nopre] skip=!isempty(VERSION.prerelease) begin
    using ADTypes

    # Only run these tests on non-prerelease Julia versions
    @info "Running Enzyme tests (Julia $(VERSION))"
    
    cache = zeros(2)
    function f(du, u, p)
        cache .= u .* u
        du .= cache .- 2
    end
    u0 = [1.0, 1.0]
    probN = NonlinearProblem{true}(f, u0)

    # Test with AutoEnzyme for autodiff
    if isempty(VERSION.prerelease)
        using Enzyme
        sol = solve(probN, RobustMultiNewton(; autodiff = AutoEnzyme()))
        @test SciMLBase.successful_retcode(sol)
        
        sol = solve(
            probN, FastShortcutNonlinearPolyalg(; autodiff = AutoEnzyme()); abstol = 1e-9
        )
        @test SciMLBase.successful_retcode(sol)
    end
end

@testitem "ForwardDiff with Enzyme backend" tags=[:nopre] skip=!isempty(VERSION.prerelease) begin
    using ForwardDiff, ADTypes
    
    # Only run these tests on non-prerelease Julia versions
    @info "Running ForwardDiff-Enzyme integration tests (Julia $(VERSION))"
    
    test_f!(du, u, p) = (@. du = u^2 - p)
    test_f(u, p) = (@. u^2 - p)
    
    function solve_oop(p)
        solve(NonlinearProblem(test_f, 2.0, p), NewtonRaphson(; autodiff = AutoEnzyme())).u
    end
    
    if isempty(VERSION.prerelease)
        using Enzyme
        
        # Test scalar AD with Enzyme backend
        for p in 1.0:0.1:10.0
            sol = solve(NonlinearProblem(test_f, 2.0, p), NewtonRaphson(; autodiff = AutoEnzyme()))
            if SciMLBase.successful_retcode(sol)
                gs = abs.(ForwardDiff.derivative(solve_oop, p))
                gs_true = abs.(1 / (2 * √p))
                @test abs.(gs) ≈ abs.(gs_true) atol=1e-5
            end
        end
    end
end