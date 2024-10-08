using TestItemRunner, InteractiveUtils, Pkg, Test

@info sprint(InteractiveUtils.versioninfo)

const GROUP = lowercase(get(ENV, "GROUP", "All"))

(GROUP == "all" || GROUP == "cuda") && Pkg.add(["CUDA"])
(GROUP == "all" || GROUP == "adjoint") && Pkg.add(["SciMLSensitivity"])
(GROUP == "all" || GROUP == "alloc_check") && Pkg.add(["AllocCheck"])

@testset "SimpleNonlinearSolve.jl" begin
    if GROUP == "all"
        @run_package_tests
    else
        @run_package_tests filter = ti -> (Symbol(GROUP) in ti.tags)
    end
end
