mutable struct ImplicitDiscreteState{uType, pType, tType}
    u::Vector{uType}
    p::Union{Nothing, Vector{pType}}
    t_next::tType
end

mutable struct IDSolveCache{uType} <: OrdinaryDiffEqMutableCache
    u::uType
    uprev::uType
    state::ImplicitDiscreteState
    prob::Union{Nothing, AbstractNonlinearProblem}
end

function alg_cache(alg::IDSolve, u, rate_prototype, ::Type{uEltypeNoUnits},
        ::Type{uBottomEltypeNoUnits}, ::Type{tTypeNoUnits}, uprev, uprev2, f, t,
        dt, reltol, p, calck,
        ::Val{true}) where {uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits}

    state = ImplicitDiscreteState(similar(u), similar(p), t)
    IDSolveCache(u, uprev, state, nothing)
end

isdiscretecache(cache::IDSolveCache) = true

function alg_cache(alg::IDSolve, u, rate_prototype, ::Type{uEltypeNoUnits},
        ::Type{uBottomEltypeNoUnits}, ::Type{tTypeNoUnits}, uprev, uprev2, f, t,
        dt, reltol, p, calck,
        ::Val{false}) where {uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits}

    state = ImplicitDiscreteState(similar(u), similar(p), t)
    IDSolveCache(u, uprev, state, nothing)
end

isfsal(alg::IDSolve) = false
get_fsalfirstlast(cache::IDSolveCache, rate_prototype) = (nothing, nothing)
