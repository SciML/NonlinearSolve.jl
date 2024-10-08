using TestItemRunner, InteractiveUtils, Test

@info sprint(InteractiveUtils.versioninfo)

@testset "BracketingNonlinearSolve.jl" begin
    @run_package_tests
end
