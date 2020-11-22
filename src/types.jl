struct NullParameters end

struct NonlinearProblem{uType,isinplace,P,F,K} <: AbstractNonlinearProblem{uType,isinplace}
    f::F
    u0::uType
    p::P
    kwargs::K
    @add_kwonly function NonlinearProblem{iip}(f,u0,p=NullParameters();kwargs...) where iip
        new{typeof(u0),iip,typeof(p),typeof(f),typeof(kwargs)}(f,u0,p,kwargs)
    end
end

NonlinearProblem(f,u0,args...;kwargs...) = NonlinearProblem{isinplace(f, 3)}(f,u0,args...;kwargs...)

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

struct BracketingSolution{uType}
    left::uType
    right::uType
    retcode::Retcode
end

struct NewtonSolution{uType}
    u::uType
    retcode::Retcode
end

function sync_residuals!(solver::BracketingImmutableSolver)
    @set! solver.fl = solver.f(solver.left, solver.p)
    @set! solver.fr = solver.f(solver.right, solver.p)
    solver
end

getsolution(sol::NewtonSolution) = sol.u
getsolution(sol::BracketingSolution) = sol.left
