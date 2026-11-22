using AirspeedVelocity, Pkg, TOML

function benchmark_project(checkout)
    paths = [realpath(checkout)]
    for path in paths
        project = TOML.parsefile(joinpath(path, "Project.toml"))
        deps = keys(get(project, "deps", ()))
        for (name, source) in get(project, "sources", ())
            name in deps && haskey(source, "path") || continue
            dep = realpath(joinpath(path, source["path"]))
            dep in paths || push!(paths, dep)
        end
    end
    length(paths) == 1 && return nothing
    env = joinpath(mktempdir(), "BenchmarkEnvironment")
    Pkg.generate(env)
    project_file = joinpath(env, "Project.toml")
    # Pkg 1.10 needs a named project to share a manifest. AirspeedVelocity copies
    # the project file, so both its source and its manifest need absolute paths.
    project = TOML.parsefile(project_file)
    project["path"] = env
    project["manifest"] = joinpath(env, "Manifest.toml")
    open(io -> TOML.print(io, project), project_file, "w")
    Pkg.activate(project_file) do
        Pkg.develop([PackageSpec(; path) for path in paths[2:end]])
    end
    return project_file
end

function benchmark_revisions(repo, revisions, output; script = nothing, kwargs...)
    repo, output = abspath(repo), abspath(output)
    revisions = String.(revisions)
    scratch = joinpath(repo, ".benchmark", "checkouts")
    mkpath(scratch)
    mkpath(output)
    checkouts = map(revisions) do revision
        path = mktempdir(scratch)
        run(`git clone --shared --no-checkout $repo $path`)
        run(`git -C $path checkout --detach $revision`)
        path
    end
    package = TOML.parsefile(joinpath(first(checkouts), "Project.toml"))["name"]
    script = script === nothing ? joinpath(first(checkouts), "benchmark", "benchmarks.jl") : abspath(script)
    for (revision, checkout) in zip(revisions, checkouts)
        project_toml = VERSION < v"1.11" ? benchmark_project(checkout) : nothing
        AirspeedVelocity.benchmark(
            PackageSpec(; name = package, path = checkout, rev = revision);
            script, project_toml, output_dir = output, extra_pkgs = ["StableRNGs"], kwargs...
        )
    end
    return AirspeedVelocity.load_results(package, revisions; input_dir = output)
end

function write_benchmark_summary(io, results)
    println(io, "## Benchmark results (Julia $(VERSION))\n")
    for (label, key) in (("Time", "median"), ("Memory", "memory"))
        println(io, "### $label\n")
        println(io, AirspeedVelocity.create_table(results; add_ratio_col = true, key))
        println(io)
    end
    return
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 3 || error("Usage: ci.jl BASE_SHA HEAD_SHA OUTPUT_DIR")
    results = benchmark_revisions(pwd(), ARGS[1:2], ARGS[3])
    if haskey(ENV, "GITHUB_STEP_SUMMARY")
        open(ENV["GITHUB_STEP_SUMMARY"], "a") do io
            write_benchmark_summary(io, results)
        end
    else
        write_benchmark_summary(stdout, results)
    end
end
