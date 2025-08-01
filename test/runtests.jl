using NonlinearSolve, Hwloc, InteractiveUtils, Pkg
using SafeTestsets
using ReTestItems

@info sprint(InteractiveUtils.versioninfo)

const GROUP = lowercase(get(ENV, "GROUP", "All"))

function activate_trim_env!()
    Pkg.activate(abspath(joinpath(dirname(@__FILE__), "trim")))
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    Pkg.instantiate()
    return nothing
end

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

if GROUP != "trim"
    ReTestItems.runtests(
        NonlinearSolve; tags = (GROUP == "all" ? nothing : [Symbol(GROUP)]),
        nworkers = RETESTITEMS_NWORKERS, nworker_threads = RETESTITEMS_NWORKER_THREADS,
        testitem_timeout = 3600
    )
elseif GROUP == "trim" && VERSION >= v"1.12.0-rc1"  # trimming has been introduced in julia 1.12
    activate_trim_env!()
    @safetestset "Clean implementation (non-trimmable)" begin
        using SciMLBase: successful_retcode
        include("trim/clean_optimization.jl")
        @test successful_retcode(minimize(1.0).retcode)
    end
    @safetestset "Trimmable implementation" begin
        using SciMLBase: successful_retcode
        include("trim/trimmable_optimization.jl")
        @test successful_retcode(minimize(1.0).retcode)
    end
end
