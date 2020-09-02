mutable struct BracketingSolver{fType, algType, uType, resType, pType, cacheType, solType}
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
