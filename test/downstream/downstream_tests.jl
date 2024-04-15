using Pkg
using SafeTestsets

function activate_downstream_env()
    Pkg.activate("downstream")
    Pkg.develop(PackageSpec(path = dirname(dirname(@__DIR__))))
    Pkg.instantiate()
end

activate_downstream_env()
@safetestset "Cache indexing test" include("cache_indexing.jl")
