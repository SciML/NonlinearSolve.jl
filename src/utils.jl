const DEFAULT_NORM = DiffEqBase.NONLINEARSOLVE_DEFAULT_NORM

@concrete mutable struct FakeLinearSolveJLCache
    A
    b
end

@concrete struct FakeLinearSolveJLResult
    cache
    u
end

# Ignores NaN
function __findmin(f, x)
    return findmin(x) do xᵢ
        fx = f(xᵢ)
        return isnan(fx) ? Inf : fx
    end
end

struct NonlinearSolveTag end

function ForwardDiff.checktag(::Type{<:ForwardDiff.Tag{<:NonlinearSolveTag, <:T}}, f::F,
        x::AbstractArray{T}) where {T, F}
    return true
end

"""
    value_derivative(f, x)

Compute `f(x), d/dx f(x)` in the most efficient way.
"""
function value_derivative(f::F, x::R) where {F, R}
    T = typeof(ForwardDiff.Tag(f, R))
    out = f(ForwardDiff.Dual{T}(x, one(x)))
    ForwardDiff.value(out), ForwardDiff.extract_derivative(T, out)
end

@inline value(x) = x
@inline value(x::Dual) = ForwardDiff.value(x)
@inline value(x::AbstractArray{<:Dual}) = map(ForwardDiff.value, x)

@inline _vec(v) = vec(v)
@inline _vec(v::Number) = v
@inline _vec(v::AbstractVector) = v

@inline _restructure(y, x) = restructure(y, x)
@inline _restructure(y::Number, x::Number) = x

DEFAULT_PRECS(W, du, u, p, t, newW, Plprev, Prprev, cachedata) = nothing, nothing

function dolinsolve(cache, precs::P, linsolve::FakeLinearSolveJLCache; A = nothing,
        linu = nothing, b = nothing, du = nothing, p = nothing, weight = nothing,
        cachedata = nothing, reltol = nothing, reuse_A_if_factorization = false) where {P}
    # Update Statistics
    cache.stats.nsolve += 1
    cache.stats.nfactors += !(A isa Number)

    A !== nothing && (linsolve.A = A)
    b !== nothing && (linsolve.b = b)
    linres = linsolve.A \ linsolve.b
    return FakeLinearSolveJLResult(linsolve, linres)
end

function dolinsolve(cache, precs::P, linsolve; A = nothing, linu = nothing, b = nothing,
        du = nothing, p = nothing, weight = nothing, cachedata = nothing, reltol = nothing,
        reuse_A_if_factorization = false) where {P}
    # Update Statistics
    cache.stats.nsolve += 1
    cache.stats.nfactors += 1

    # Some Algorithms would reuse factorization but it causes the cache to not reset in
    # certain cases
    if A !== nothing
        alg = __getproperty(linsolve, Val(:alg))
        if alg !== nothing && ((alg isa LinearSolve.AbstractFactorization) ||
            (alg isa LinearSolve.DefaultLinearSolver && !(alg ==
               LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.KrylovJL_GMRES))))
            # Factorization Algorithm
            if reuse_A_if_factorization
                cache.stats.nfactors -= 1
            else
                linsolve.A = A
            end
        else
            linsolve.A = A
        end
    else
        cache.stats.nfactors -= 1
    end
    b !== nothing && (linsolve.b = b)
    linu !== nothing && (linsolve.u = linu)

    Plprev = linsolve.Pl isa ComposePreconditioner ? linsolve.Pl.outer : linsolve.Pl
    Prprev = linsolve.Pr isa ComposePreconditioner ? linsolve.Pr.outer : linsolve.Pr

    _Pl, _Pr = precs(linsolve.A, du, linu, p, nothing, A !== nothing, Plprev, Prprev,
        cachedata)
    if (_Pl !== nothing || _Pr !== nothing)
        _weight = weight === nothing ?
                  (linsolve.Pr isa Diagonal ? linsolve.Pr.diag : linsolve.Pr.inner.diag) :
                  weight
        Pl, Pr = wrapprecs(_Pl, _Pr, _weight)
        linsolve.Pl = Pl
        linsolve.Pr = Pr
    end

    linres = reltol === nothing ? solve!(linsolve) : solve!(linsolve; reltol)

    return linres
end

function wrapprecs(_Pl, _Pr, weight)
    if _Pl !== nothing
        Pl = ComposePreconditioner(InvPreconditioner(Diagonal(_vec(weight))), _Pl)
    else
        Pl = InvPreconditioner(Diagonal(_vec(weight)))
    end

    if _Pr !== nothing
        Pr = ComposePreconditioner(Diagonal(_vec(weight)), _Pr)
    else
        Pr = Diagonal(_vec(weight))
    end

    return Pl, Pr
end

concrete_jac(_) = nothing
concrete_jac(::AbstractNewtonAlgorithm{CJ}) where {CJ} = CJ

_mutable_zero(x) = zero(x)
_mutable_zero(x::SArray) = MArray(x)

_mutable(x) = x
_mutable(x::SArray) = MArray(x)

# __maybe_mutable(x, ::AbstractFiniteDifferencesMode) = _mutable(x)
# The shadow allocated for Enzyme needs to be mutable
__maybe_mutable(x, ::AutoSparseEnzyme) = _mutable(x)
__maybe_mutable(x, _) = x

# Helper function to get value of `f(u, p)`
function evaluate_f(prob::Union{NonlinearProblem{uType, iip},
            NonlinearLeastSquaresProblem{uType, iip}}, u) where {uType, iip}
    @unpack f, u0, p = prob
    if iip
        fu = f.resid_prototype === nothing ? similar(u) : f.resid_prototype
        f(fu, u, p)
    else
        fu = f(u, p)
    end
    return fu
end

function evaluate_f(f::F, u, p, ::Val{iip}; fu = nothing) where {F, iip}
    if iip
        f(fu, u, p)
        return fu
    else
        return f(u, p)
    end
end

function evaluate_f(cache::AbstractNonlinearSolveCache, u, p,
        fu_sym::Val{FUSYM} = Val(nothing)) where {FUSYM}
    cache.stats.nf += 1
    if FUSYM === nothing
        if isinplace(cache)
            cache.prob.f(get_fu(cache), u, p)
        else
            set_fu!(cache, cache.prob.f(u, p))
        end
    else
        if isinplace(cache)
            cache.prob.f(__getproperty(cache, fu_sym), u, p)
        else
            setproperty!(cache, FUSYM, cache.prob.f(u, p))
        end
    end
    return nothing
end

# Concretize Algorithms
function get_concrete_algorithm(alg, prob)
    !hasfield(typeof(alg), :ad) && return alg
    alg.ad isa ADTypes.AbstractADType && return alg

    # Figure out the default AD
    # Now that we have handed trivial cases, we can allow extending this function
    # for specific algorithms
    return __get_concrete_algorithm(alg, prob)
end

function __get_concrete_algorithm(alg, prob)
    @unpack sparsity, jac_prototype = prob.f
    use_sparse_ad = sparsity !== nothing || jac_prototype !== nothing
    ad = if !ForwardDiff.can_dual(eltype(prob.u0))
        # Use Finite Differencing
        use_sparse_ad ? AutoSparseFiniteDiff() : AutoFiniteDiff()
    else
        (use_sparse_ad ? AutoSparseForwardDiff : AutoForwardDiff)(;
            tag = ForwardDiff.Tag(NonlinearSolveTag(), eltype(prob.u0)))
    end
    return set_ad(alg, ad)
end

function init_termination_cache(abstol, reltol, du, u, ::Nothing)
    return init_termination_cache(abstol, reltol, du, u, AbsSafeBestTerminationMode())
end
function init_termination_cache(abstol, reltol, du, u, tc::AbstractNonlinearTerminationMode)
    tc_cache = init(du, u, tc; abstol, reltol)
    return DiffEqBase.get_abstol(tc_cache), DiffEqBase.get_reltol(tc_cache), tc_cache
end

function check_and_update!(cache, fu, u, uprev)
    return check_and_update!(cache.tc_cache, cache, fu, u, uprev)
end
function check_and_update!(tc_cache, cache, fu, u, uprev)
    return check_and_update!(tc_cache, cache, fu, u, uprev,
        DiffEqBase.get_termination_mode(tc_cache))
end
function check_and_update!(tc_cache, cache, fu, u, uprev,
        mode::AbstractNonlinearTerminationMode)
    if tc_cache(fu, u, uprev)
        # Just a sanity measure!
        if isinplace(cache)
            cache.prob.f(get_fu(cache), u, cache.prob.p)
        else
            set_fu!(cache, cache.prob.f(u, cache.prob.p))
        end
        cache.force_stop = true
    end
end
function check_and_update!(tc_cache, cache, fu, u, uprev,
        mode::AbstractSafeNonlinearTerminationMode)
    if tc_cache(fu, u, uprev)
        if tc_cache.retcode == NonlinearSafeTerminationReturnCode.Success
            cache.retcode = ReturnCode.Success
        end
        if tc_cache.retcode == NonlinearSafeTerminationReturnCode.PatienceTermination
            cache.retcode = ReturnCode.ConvergenceFailure
        end
        if tc_cache.retcode == NonlinearSafeTerminationReturnCode.ProtectiveTermination
            cache.retcode = ReturnCode.Unstable
        end
        # Just a sanity measure!
        if isinplace(cache)
            cache.prob.f(get_fu(cache), u, cache.prob.p)
        else
            set_fu!(cache, cache.prob.f(u, cache.prob.p))
        end
        cache.force_stop = true
    end
end
function check_and_update!(tc_cache, cache, fu, u, uprev,
        mode::AbstractSafeBestNonlinearTerminationMode)
    if tc_cache(fu, u, uprev)
        if tc_cache.retcode == NonlinearSafeTerminationReturnCode.Success
            cache.retcode = ReturnCode.Success
        end
        if tc_cache.retcode == NonlinearSafeTerminationReturnCode.PatienceTermination
            cache.retcode = ReturnCode.ConvergenceFailure
        end
        if tc_cache.retcode == NonlinearSafeTerminationReturnCode.ProtectiveTermination
            cache.retcode = ReturnCode.Unstable
        end
        if isinplace(cache)
            copyto!(get_u(cache), tc_cache.u)
            cache.prob.f(get_fu(cache), get_u(cache), cache.prob.p)
        else
            set_u!(cache, tc_cache.u)
            set_fu!(cache, cache.prob.f(get_u(cache), cache.prob.p))
        end
        cache.force_stop = true
    end
end

@inline __init_identity_jacobian(u::Number, fu, α = true) = oftype(u, α)
@inline @views function __init_identity_jacobian(u, fu, α = true)
    J = similar(fu, promote_type(eltype(fu), eltype(u)), length(fu), length(u))
    fill!(J, zero(eltype(J)))
    if fast_scalar_indexing(J)
        @inbounds for i in axes(J, 1)
            J[i, i] = α
        end
    else
        J[diagind(J)] .= α
    end
    return J
end
@inline function __init_identity_jacobian(u::StaticArray, fu::StaticArray, α = true)
    T = promote_type(eltype(fu), eltype(u))
    return MArray{Tuple{prod(Size(fu)), prod(Size(u))}, T}(I * α)
end
@inline function __init_identity_jacobian(u::SArray, fu::SArray, α = true)
    T = promote_type(eltype(fu), eltype(u))
    return SArray{Tuple{prod(Size(fu)), prod(Size(u))}, T}(I * α)
end

@inline __reinit_identity_jacobian!!(J::Number, α = true) = oftype(J, α)
@inline __reinit_identity_jacobian!!(J::AbstractVector, α = true) = fill!(J, α)
@inline @views function __reinit_identity_jacobian!!(J::AbstractMatrix, α = true)
    fill!(J, zero(eltype(J)))
    if fast_scalar_indexing(J)
        @inbounds for i in axes(J, 1)
            J[i, i] = α
        end
    else
        J[diagind(J)] .= α
    end
    return J
end
@inline function __reinit_identity_jacobian!!(J::SVector, α = true)
    return ones(SArray{Tuple{Size(J)[1]}, eltype(J)}) .* α
end
@inline function __reinit_identity_jacobian!!(J::SMatrix, α = true)
    S = Size(J)
    return SArray{Tuple{S[1], S[2]}, eltype(J)}(I) .* α
end

function __init_low_rank_jacobian(u::StaticArray{S1, T1}, fu::StaticArray{S2, T2},
        ::Val{threshold}) where {S1, S2, T1, T2, threshold}
    T = promote_type(T1, T2)
    fuSize, uSize = Size(fu), Size(u)
    Vᵀ = MArray{Tuple{threshold, prod(uSize)}, T}(undef)
    U = MArray{Tuple{prod(fuSize), threshold}, T}(undef)
    return U, Vᵀ
end
function __init_low_rank_jacobian(u, fu, ::Val{threshold}) where {threshold}
    Vᵀ = similar(u, threshold, length(u))
    U = similar(u, length(fu), threshold)
    return U, Vᵀ
end

@inline __is_ill_conditioned(x::Number) = iszero(x)
@inline __is_ill_conditioned(x::AbstractMatrix) = cond(x) ≥
                                                  inv(eps(real(eltype(x)))^(1 // 2))
@inline __is_ill_conditioned(x::AbstractVector) = any(iszero, x)
@inline __is_ill_conditioned(x) = false

# Safe getproperty
@generated function __getproperty(s::S, ::Val{X}) where {S, X}
    hasfield(S, X) && return :(s.$X)
    return :(nothing)
end

# Non-square matrix
@inline __needs_square_A(_, ::Number) = true
@inline __needs_square_A(alg, _) = LinearSolve.needs_square_A(alg.linsolve)

# Define special concatenation for certain Array combinations
@inline _vcat(x, y) = vcat(x, y)

# LazyArrays for tracing
__zero(x::AbstractArray) = zero(x)
__zero(x) = x
LazyArrays.applied_eltype(::typeof(__zero), x) = eltype(x)
LazyArrays.applied_ndims(::typeof(__zero), x) = ndims(x)
LazyArrays.applied_size(::typeof(__zero), x) = size(x)
LazyArrays.applied_axes(::typeof(__zero), x) = axes(x)

# Safe Inverse: Try to use `inv` but if lu fails use `pinv`
@inline __safe_inv(A::Number) = pinv(A)
@inline __safe_inv(A::AbstractMatrix) = pinv(A)
@inline __safe_inv(A::AbstractVector) = __safe_inv(Diagonal(A)).diag
@inline __safe_inv(A::ApplyArray) = __safe_inv(A.f(A.args...))
@inline function __safe_inv(A::StridedMatrix{T}) where {T}
    LinearAlgebra.checksquare(A)
    if istriu(A)
        A_ = UpperTriangular(A)
        issingular = any(iszero, @view(A_[diagind(A_)]))
        !issingular && return triu!(parent(inv(A_)))
    elseif istril(A)
        A_ = LowerTriangular(A)
        issingular = any(iszero, @view(A_[diagind(A_)]))
        !issingular && return tril!(parent(inv(A_)))
    else
        F = lu(A; check = false)
        if issuccess(F)
            Ai = LinearAlgebra.inv!(F)
            return convert(typeof(parent(Ai)), Ai)
        end
    end
    return pinv(A)
end
@inline __safe_inv(A::SparseMatrixCSC) = __safe_inv(Matrix(A))

LazyArrays.applied_eltype(::typeof(__safe_inv), x) = eltype(x)
LazyArrays.applied_ndims(::typeof(__safe_inv), x) = ndims(x)
LazyArrays.applied_size(::typeof(__safe_inv), x) = size(x)
LazyArrays.applied_axes(::typeof(__safe_inv), x) = axes(x)

# SparseAD --> NonSparseAD
@inline __get_nonsparse_ad(::AutoSparseForwardDiff) = AutoForwardDiff()
@inline __get_nonsparse_ad(::AutoSparseFiniteDiff) = AutoFiniteDiff()
@inline __get_nonsparse_ad(::AutoSparseZygote) = AutoZygote()
@inline __get_nonsparse_ad(ad) = ad

# Use Symmetric Matrices if known to be efficient
@inline __maybe_symmetric(x) = Symmetric(x)
@inline __maybe_symmetric(x::Number) = x
## LinearSolve with `nothing` doesn't dispatch correctly here
@inline __maybe_symmetric(x::StaticArray) = x
@inline __maybe_symmetric(x::SparseArrays.AbstractSparseMatrix) = x
@inline __maybe_symmetric(x::SciMLOperators.AbstractSciMLOperator) = x

# Unalias
@inline __maybe_unaliased(x::Union{Number, SArray}, ::Bool) = x
@inline function __maybe_unaliased(x::AbstractArray, alias::Bool)
    # Spend time coping iff we will mutate the array
    (alias || !can_setindex(typeof(x))) && return x
    return deepcopy(x)
end

# Init ones
@inline function __init_ones(x)
    w = similar(x)
    recursivefill!(w, true)
    return w
end
@inline __init_ones(x::StaticArray) = ones(typeof(x))

# Diagonal of type `u`
__init_diagonal(u::Number, v) = oftype(u, v)
function __init_diagonal(u::SArray, v)
    u_ = vec(u)
    return Diagonal(ones(typeof(u_)) * v)
end
function __init_diagonal(u, v)
    d = similar(vec(u))
    d .= v
    return Diagonal(d)
end

# Reduce sum
function __sum_JᵀJ!!(y, J)
    if setindex_trait(y) === CanSetindex()
        sum!(abs2, y, J')
        return y
    else
        return sum(abs2, J'; dims = 1)
    end
end

# Alpha for Initial Jacobian Guess
# The values are somewhat different from SciPy, these were tuned to the 23 test problems
@inline function __initial_inv_alpha(α::Number, u, fu, norm::F) where {F}
    return convert(promote_type(eltype(u), eltype(fu)), inv(α))
end
@inline function __initial_inv_alpha(::Nothing, u, fu, norm::F) where {F}
    norm_fu = norm(fu)
    return ifelse(norm_fu ≥ 1e-5, max(norm(u), true) / (2 * norm_fu),
        convert(promote_type(eltype(u), eltype(fu)), true))
end
@inline __initial_inv_alpha(inv_α, α::Number, u, fu, norm::F) where {F} = inv_α
@inline function __initial_inv_alpha(inv_α, α::Nothing, u, fu, norm::F) where {F}
    return __initial_inv_alpha(α, u, fu, norm)
end

@inline function __initial_alpha(α::Number, u, fu, norm::F) where {F}
    return convert(promote_type(eltype(u), eltype(fu)), α)
end
@inline function __initial_alpha(::Nothing, u, fu, norm::F) where {F}
    norm_fu = norm(fu)
    return ifelse(1e-5 ≤ norm_fu ≤ 1e5, max(norm(u), true) / (2 * norm_fu),
        convert(promote_type(eltype(u), eltype(fu)), true))
end
@inline __initial_alpha(α_initial, α::Number, u, fu, norm::F) where {F} = α_initial
@inline function __initial_alpha(α_initial, α::Nothing, u, fu, norm::F) where {F}
    return __initial_alpha(α, u, fu, norm)
end

# Diagonal
@inline function __get_diagonal!!(J::AbstractVector, J_full::AbstractMatrix)
    if can_setindex(J)
        if fast_scalar_indexing(J)
            @inbounds for i in eachindex(J)
                J[i] = J_full[i, i]
            end
        else
            J .= view(J_full, diagind(J_full))
        end
    else
        J = __diag(J_full)
    end
    return J
end
@inline function __get_diagonal!!(J::AbstractArray, J_full::AbstractMatrix)
    return _restructure(J, __get_diagonal!!(_vec(J), J_full))
end
@inline __get_diagonal!!(J::Number, J_full::Number) = J_full

@inline __diag(x::AbstractMatrix) = diag(x)
@inline __diag(x::AbstractVector) = x
@inline __diag(x::Number) = x

#functions for updating alpha for PseudoTransient
function switched_evolution_relaxation(alpha::Number,
        res_norm::Number,
        nsteps::Int,
        u,
        u_prev,
        fu,
        norm::F) where {F}
    new_norm = norm(fu)
    return alpha * (res_norm / new_norm)
end

function robust_update_alpha(alpha::Number,
        res_norm::Number,
        nsteps::Int,
        u,
        u_prev,
        fu,
        norm::F) where {F}
    if nsteps ≤ 50
        return alpha
    else
        new_norm = norm(fu)
        return alpha * (res_norm / new_norm)
    end
end

@inline __is_complex(::Type{ComplexF64}) = true
@inline __is_complex(::Type{ComplexF32}) = true
@inline __is_complex(::Type{Complex}) = true
@inline __is_complex(::Type{T}) where {T} = false
