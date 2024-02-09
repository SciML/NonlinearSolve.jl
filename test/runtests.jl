using ReTestItems

const GROUP = get(ENV, "GROUP", "All")

if GROUP == "All" || GROUP == "Core"
    ReTestItems.runtests(joinpath(@__DIR__, "core/"),
        joinpath(@__DIR__, "misc/"),
        joinpath(@__DIR__, "wrappers/"))
end

if GROUP == "GPU"
    ReTestItems.runtests(joinpath(@__DIR__, "gpu/"))
end
