@enum Retcode::Int begin
    DEFAULT
    EXACT_SOLUTION_LEFT
    EXACT_SOLUTION_RIGHT
    MAXITERS_EXCEED
    FLOATING_POINT_LIMIT
end

struct BracketingImmutableSolver{fType, algType, uType, resType, pType, cacheType} <: AbstractImmutableNonlinearSolver
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
    retcode::Retcode
    cache::cacheType
    iip::Bool
end

# function BracketingImmutableSolver(iip, iter, f, alg, left, right, fl, fr, p, force_stop, maxiters, retcode, cache)
#     BracketingImmutableSolver{iip, typeof(f), typeof(alg),
#         typeof(left), typeof(fl), typeof(p), typeof(cache)}(iter, f, alg, left, right, fl, fr, p, force_stop, maxiters, retcode, cache)
# end

struct NewtonImmutableSolver{fType, algType, uType, resType, pType, INType, tolType, cacheType} <: AbstractImmutableNonlinearSolver
    iter::Int
    f::fType
    alg::algType
    u::uType
    fu::resType
    p::pType
    force_stop::Bool
    maxiters::Int
    internalnorm::INType
    retcode::Retcode
    tol::tolType
    cache::cacheType
    iip::Bool
end

# function NewtonImmutableSolver{iip}(iter, f, alg, u, fu, p, force_stop, maxiters, internalnorm, retcode, tol, cache) where iip
#     NewtonImmutableSolver{iip, typeof(f), typeof(alg), typeof(u),
#         typeof(fu), typeof(p), typeof(internalnorm), typeof(tol), typeof(cache)}(iter, f, alg, u, fu, p, force_stop, maxiters, internalnorm, retcode, tol, cache)
# end

struct BracketingSolution{uType,resType}
    left::uType
    right::uType
    retcode::Retcode
    resid::resType
end

struct NewtonSolution{uType,resType}
    u::uType
    retcode::Retcode
    resid::resType
end

function sync_residuals!(solver::BracketingImmutableSolver)
    @set! solver.fl = solver.f(solver.left, solver.p)
    @set! solver.fr = solver.f(solver.right, solver.p)
    solver
end

getsolution(sol::NewtonSolution) = sol.u
getsolution(sol::BracketingSolution) = sol.left
