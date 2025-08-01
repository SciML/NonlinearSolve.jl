using ReTestItems, Hwloc, InteractiveUtils, Pkg

@info sprint(InteractiveUtils.versioninfo)

const GROUP = lowercase(get(ENV, "GROUP", "All"))

if GROUP != "trim"
    using NonlinearSolve  # trimming uses a NonlinearSolve from a custom environment

    const EXTRA_PKGS = Pkg.PackageSpec[]
    if GROUP == "all" || GROUP == "downstream"
        push!(EXTRA_PKGS, Pkg.PackageSpec("ModelingToolkit"))
        push!(EXTRA_PKGS, Pkg.PackageSpec("SymbolicIndexingInterface"))
    end
    length(EXTRA_PKGS) ≥ 1 && Pkg.add(EXTRA_PKGS)

    const RETESTITEMS_NWORKERS = parse(
        Int, get(
            ENV, "RETESTITEMS_NWORKERS",
            string(min(ifelse(Sys.iswindows(), 0, Hwloc.num_physical_cores()), 4))
        )
    )
    const RETESTITEMS_NWORKER_THREADS = parse(
        Int,
        get(
            ENV, "RETESTITEMS_NWORKER_THREADS",
            string(max(Hwloc.num_virtual_cores() ÷ max(RETESTITEMS_NWORKERS, 1), 1))
        )
    )

    @info "Running tests for group: $(GROUP) with $(RETESTITEMS_NWORKERS) workers"

    ReTestItems.runtests(
        NonlinearSolve; tags = (GROUP == "all" ? nothing : [Symbol(GROUP)]),
        nworkers = RETESTITEMS_NWORKERS, nworker_threads = RETESTITEMS_NWORKER_THREADS,
        testitem_timeout = 3600
    )
elseif GROUP == "trim" && VERSION >= v"1.12.0-rc1"  # trimming has been introduced in julia 1.12
    Pkg.activate(joinpath(dirname(@__FILE__), "trim"))
    Pkg.instantiate()
    include("trim/runtests.jl")
end
