mutable struct BracketingSolver{fType, algType, uType, resType, pType, cacheType} <: AbstractNonlinearSolver
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
end

struct BracketingImmutableSolver{fType, algType, uType, resType, pType} <: AbstractImmutableNonlinearSolver
    iter::Int
    f::fType
    alg::algType
    left::uType
    right::uType
    fl::resType
    fr::resType
    p::pType
    force_stop::Bool
    maxiters::Int
    retcode::Symbol
end


mutable struct NewtonSolver{fType, algType, uType, resType, pType, cacheType, INType, tolType} <: AbstractNonlinearSolver
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
end

struct NewtonImmutableSolver{fType, algType, uType, resType, pType, INType, tolType} <: AbstractImmutableNonlinearSolver
    iter::Int
    f::fType
    alg::algType
    u::uType
    fu::resType
    p::pType
    force_stop::Bool
    maxiters::Int
    internalnorm::INType
    retcode::Symbol
    tol::tolType
end

struct BracketingSolution{uType}
    left::uType
    right::uType
    retcode::Symbol
end

struct NewtonSolution{uType}
    u::uType
    retcode::Symbol
end

function sync_residuals!(solver::BracketingSolver)
    solver.fl = solver.f(solver.left, solver.p)
    solver.fr = solver.f(solver.right, solver.p)
    nothing
end
