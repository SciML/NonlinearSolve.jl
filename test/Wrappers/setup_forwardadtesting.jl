using Reexport, NonlinearSolve
@reexport using ForwardDiff, MINPACK, NLsolve, StaticArrays, Sundials, LinearAlgebra

test_f!(du, u, p) = (@. du = u^2 - p)
test_f(u, p) = (@. u^2 - p)

jacobian_f(::Number, p) = 1 / (2 * √p)
jacobian_f(::Number, p::Number) = 1 / (2 * √p)
jacobian_f(u, p::Number) = one.(u) .* (1 / (2 * √p))
jacobian_f(u, p::AbstractArray) = diagm(vec(@. 1 / (2 * √p)))

function solve_with(::Val{mode}, u, alg) where {mode}
    f = if mode === :iip
        solve_iip(p) = solve(NonlinearProblem(test_f!, u, p), alg).u
    elseif mode === :iip_cache
        function solve_iip_init(p)
            cache = SciMLBase.init(NonlinearProblem(test_f!, u, p), alg)
            return SciMLBase.solve!(cache).u
        end
    elseif mode === :oop
        solve_oop(p) = solve(NonlinearProblem(test_f, u, p), alg).u
    elseif mode === :oop_cache
        function solve_oop_init(p)
            cache = SciMLBase.init(NonlinearProblem(test_f, u, p), alg)
            return SciMLBase.solve!(cache).u
        end
    end
    return f
end

compatible(::Any, ::Val{:oop}) = true
compatible(::Any, ::Val{:oop_cache}) = true
compatible(::Number, ::Val{:iip}) = false
compatible(::AbstractArray, ::Val{:iip}) = true
compatible(::StaticArray, ::Val{:iip}) = false
compatible(::Number, ::Val{:iip_cache}) = false
compatible(::AbstractArray, ::Val{:iip_cache}) = true
compatible(::StaticArray, ::Val{:iip_cache}) = false

compatible(::Any, ::Number) = true
compatible(::Number, ::AbstractArray) = false
compatible(u::AbstractArray, p::AbstractArray) = size(u) == size(p)

compatible(u::Number, ::SciMLBase.AbstractNonlinearAlgorithm) = true
compatible(u::Number, ::Union{CMINPACK, NLsolveJL, KINSOL}) = true
compatible(u::AbstractArray, ::SciMLBase.AbstractNonlinearAlgorithm) = true
compatible(u::AbstractArray{T, N}, ::KINSOL) where {T, N} = N == 1  # Need to be fixed upstream
compatible(u::StaticArray{S, T, N}, ::KINSOL) where {S <: Tuple, T, N} = false
compatible(u::StaticArray, ::SciMLBase.AbstractNonlinearAlgorithm) = true
compatible(u::StaticArray, ::Union{CMINPACK, NLsolveJL, KINSOL}) = false
compatible(u, ::Nothing) = true

compatible(::Any, ::Any) = true
compatible(::CMINPACK, ::Val{:iip_cache}) = false
compatible(::CMINPACK, ::Val{:oop_cache}) = false
compatible(::NLsolveJL, ::Val{:iip_cache}) = false
compatible(::NLsolveJL, ::Val{:oop_cache}) = false
compatible(::KINSOL, ::Val{:iip_cache}) = false
compatible(::KINSOL, ::Val{:oop_cache}) = false

export test_f!, test_f, jacobian_f, solve_with, compatible
