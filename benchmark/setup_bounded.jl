using Pkg, TOML

function bounded_benchmark_environment(environment)
    root = dirname(@__DIR__)
    paths = Set{String}()
    function collect_sources(path)
        path = abspath(path)
        path in paths && return
        push!(paths, path)
        project = TOML.parsefile(joinpath(path, "Project.toml"))
        for source in values(get(project, "sources", Dict()))
            haskey(source, "path") && collect_sources(joinpath(path, source["path"]))
        end
        return nothing
    end
    collect_sources(root)
    Pkg.activate(environment)
    Pkg.develop([PackageSpec(; path) for path in sort!(collect(paths))])
    Pkg.add(["BenchmarkTools", "LinearSolve", "ADTypes", "SciMLBase", "SparseArrays", "Statistics", "ForwardDiff"])
    return nothing
end

bounded_benchmark_environment(isempty(ARGS) ? joinpath(dirname(@__DIR__), ".benchmark", "bounded") : only(ARGS))
