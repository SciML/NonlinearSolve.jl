using ReTestItems, CUDA

const GROUP = get(ENV, "GROUP", CUDA.functional() ? "All" : "Core")

if GROUP == "All"
    ReTestItems.runtests(@__DIR__)
else
    tags = [Symbol(lowercase(GROUP))]
    ReTestItems.runtests(@__DIR__; tags)
end
