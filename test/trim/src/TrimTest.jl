module TrimTest
#=
Currently, trimming only works if the target code is in a package. I.e., trying to trim
```julia
include("optimization_trimmable.jl")
function (@main)(argv::Vector{String})::Cint
    minimize(1.0)
    return 0
end
```
or even
```julia
mod MyMod
    include("optimization_trimmable.jl")
end
function (@main)(argv::Vector{String})::Cint
    MyMod.minimize(1.0)
    return 0
end
```
segfaults `juliac`. Looking at the segfault stacktrace it seems the culprit is
`const cache = init(...)`. Either way, we circumvent the segfault by putting
this below code into a package definition.
=#
include("../optimization_trimmable.jl")
include("../optimization_clean.jl")
end
