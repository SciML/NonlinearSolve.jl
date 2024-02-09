using Pkg, ReTestItems

const GROUP = get(ENV, "GROUP", "All")

function activate_env(env)
    Pkg.activate(env)
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    Pkg.instantiate()
end

if GROUP == "All" || GROUP == "Core"
    ReTestItems.runtests(joinpath(@__DIR__, "core/"),
        joinpath(@__DIR__, "misc/"),
        joinpath(@__DIR__, "wrappers/"))
end

if GROUP == "GPU"
    activate_env("gpu")
    ReTestItems.runtests(joinpath(@__DIR__, "gpu/"))
end
