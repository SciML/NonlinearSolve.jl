"""
```julia
Itp(; k1 = Val{1}(), k2 = Val{2}(), n0 = Val{1}())
```
ITP (Interpolate Truncate & Project)


"""

struct Itp <: AbstractBracketingAlgorithm
    k1::Real
    k2::Real
    n0::Int
end

function Itp(k1::Real = Val{1}(), k2::Real = Val{2}(), n0::Int = Val{1}())
    if k1 < 0
        ArgumentError("Hyper-parameter κ₁ should not be negative")
    end
    if !isa(n0, Int)
        ArgumentError("Hyper-parameter n₀ should be an Integer")
    end
    Itp(k1, k2, n0)
end

function SciMLBase.__solve(prob::NonlinearProblem, alg::Itp,
                            args..., abstol = nothing, reltol = nothing, 
                            maxiters = 1000, kwargs...)

                               
end