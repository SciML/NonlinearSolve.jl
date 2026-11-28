using Test
include("ci.jl")

@testset "LTS benchmark project resolves checkout sources after copying" begin
    checkout = dirname(@__DIR__)
    runner_project = joinpath(mktempdir(), "Project.toml")
    cp(benchmark_project(checkout), runner_project)
    Pkg.activate(runner_project) do
        packages = Dict(info.name => info for info in values(Pkg.dependencies()))
        for name in ("NonlinearSolveBase", "NonlinearSolveQuasiNewton", "SciMLJacobianOperators")
            @test realpath(packages[name].source) == realpath(joinpath(checkout, "lib", name))
        end
    end
end
