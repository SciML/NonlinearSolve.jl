using ReTestItems

const GROUP = get(ENV, "GROUP", "All")

if GROUP == "All" || GROUP == "Core"
    ReTestItems.runtests(joinpath(@__DIR__, "core/"))
end

if GROUP == "GPU"
    ReTestItems.runtests(joinpath(@__DIR__, "gpu/"))
end