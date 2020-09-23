mutable struct BracketingSolver{fType, algType, uType, resType, pType, cacheType, solType} <: AbstractNonlinearSolver
    iter::Int
    f::fType
    alg::algType
    left::uType
    right::uType
    fl::resType
    fr::resType
    p::pType
    cache::cacheType
    force_stop::Bool
    maxiters::Int
    retcode::Symbol
    sol::solType
end

struct BracketingImmutableSolver{fType, algType, uType, resType, pType, cacheType, solType} <: AbstractImmutableNonlinearSolver
    iter::Int
    f::fType
    alg::algType
    left::uType
    right::uType
    fl::resType
    fr::resType
    p::pType
    cache::cacheType
    force_stop::Bool
    maxiters::Int
    retcode::Symbol
    sol::solType
end


mutable struct NewtonSolver{fType, algType, uType, resType, pType, cacheType, INType, tolType, solType} <: AbstractNonlinearSolver
    iter::Int
    f::fType
    alg::algType
    u::uType
    fu::resType
    p::pType
    cache::cacheType
    force_stop::Bool
    maxiters::Int
    internalnorm::INType
    retcode::Symbol
    tol::tolType
    sol::solType
end

struct NewtonImmutableSolver{fType, algType, uType, resType, pType, cacheType, INType, tolType, solType} <: AbstractImmutableNonlinearSolver
    iter::Int
    f::fType
    alg::algType
    u::uType
    fu::resType
    p::pType
    cache::cacheType
    force_stop::Bool
    maxiters::Int
    internalnorm::INType
    retcode::Symbol
    tol::tolType
    sol::solType
end

function sync_residuals!(solver::BracketingSolver)
    solver.fl = solver.f(solver.left, solver.p)
    solver.fr = solver.f(solver.right, solver.p)
    nothing
end

mutable struct BracketingSolution{uType}
    left::uType
    right::uType
    retcode::Symbol
end

function build_solution(u_prototype, ::Val{true})
    return BracketingSolution(similar(u_prototype), similar(u_prototype), :Default)
end

function build_solution(u_prototype, ::Val{false})
    return BracketingSolution(zero(u_prototype), zero(u_prototype), :Default)
end

struct NewtonSolution{uType}
    u::uType
    retcode::Symbol
end

function build_newton_solution(u_prototype, ::Val{iip}) where iip
    return NewtonSolution(zero(u_prototype), :Default)
end

