"""
    LevenbergMarquardtTrustRegion(β_uphill)

Trust Region method for [`LevenbergMarquardt`](@ref). This method is tightly coupled with
the Levenberg-Marquardt method and works by directly updating the damping parameter instead
of specifying a trust region radius.

### Arguments

  - `β_uphill`: a factor that determines if a step is accepted or rejected. The standard
    choice in the Levenberg-Marquardt method is to accept all steps that decrease the cost
    and reject all steps that increase the cost. Although this is a natural and safe choice,
    it is often not the most efficient. Therefore downhill moves are always accepted, but
    uphill moves are only conditionally accepted. To decide whether an uphill move will be
    accepted at each iteration ``i``, we compute
    ``\\beta_i = \\cos(v_{\\text{new}}, v_{\\text{old}})``, which denotes the cosine angle
    between the proposed velocity ``v_{\\text{new}}`` and the velocity of the last accepted
    step ``v_{\\text{old}}``. The idea is to accept uphill moves if the angle is small. To
    specify, uphill moves are accepted if
    ``(1-\\beta_i)^{b_{\\text{uphill}}} C_{i+1} \\le C_i``, where ``C_i`` is the cost at
    iteration ``i``. Reasonable choices for `b_uphill` are `1.0` or `2.0`, with `b_uphill=2.0`
    allowing higher uphill moves than `b_uphill=1.0`. When `b_uphill=0.0`, no uphill moves
    will be accepted. Defaults to `1.0`. For more details, see section 4 of [1]
    [this paper](https://arxiv.org/abs/1201.5885).

### References

[1] Transtrum, Mark K., and James P. Sethna. "Improvements to the Levenberg-Marquardt
algorithm for nonlinear least-squares minimization." arXiv preprint arXiv:1201.5885 (2012).
"""
@concrete struct LevenbergMarquardtTrustRegion <: AbstractTrustRegionMethod
    β_uphill
end

function Base.show(io::IO, alg::LevenbergMarquardtTrustRegion)
    print(io, "LevenbergMarquardtTrustRegion(β_uphill = $(alg.β_uphill))")
end

@concrete mutable struct LevenbergMarquardtTrustRegionCache <:
                         AbstractTrustRegionMethodCache
    f
    p
    loss_old
    v_cache
    norm_v_old
    internalnorm
    β_uphill
    last_step_accepted::Bool
    u_cache
    fu_cache
    nf::Int
end

function reinit_cache!(cache::LevenbergMarquardtTrustRegionCache, args...; p = cache.p,
        u0 = cache.v_cache, kwargs...)
    cache.p = p
    @bb copyto!(cache.v_cache, u0)
    cache.loss_old = oftype(cache.loss_old, Inf)
    cache.norm_v_old = oftype(cache.norm_v_old, Inf)
    cache.last_step_accepted = false
    cache.nf = 0
end

function SciMLBase.init(prob::AbstractNonlinearProblem, alg::LevenbergMarquardtTrustRegion,
        f::F, fu, u, p, args...; internalnorm::IF = DEFAULT_NORM, kwargs...) where {F, IF}
    T = promote_type(eltype(u), eltype(fu))
    @bb v = copy(u)
    @bb u_cache = similar(u)
    @bb fu_cache = similar(fu)
    return LevenbergMarquardtTrustRegionCache(f, p, T(Inf), v, T(Inf), internalnorm,
        alg.β_uphill, false, u_cache, fu_cache, 0)
end

function SciMLBase.solve!(cache::LevenbergMarquardtTrustRegionCache, J, fu, u, δu,
        damping_stats)
    # This should be true if Geodesic Acceleration is being used
    v = hasfield(typeof(damping_stats), :v) ? damping_stats.v : δu
    norm_v = cache.internalnorm(v)
    β = dot(v, cache.v_cache) / (norm_v * cache.norm_v_old)

    @bb @. cache.u_cache = u + δu
    cache.fu_cache = evaluate_f!!(cache.f, cache.fu_cache, cache.u_cache, cache.p)
    cache.nf += 1

    loss = cache.internalnorm(cache.fu_cache)

    if (1 - β)^cache.β_uphill * loss ≤ cache.loss_old  # Accept Step
        cache.last_step_accepted = true
        cache.norm_v_old = norm_v
        @bb copyto!(cache.v_cache, v)
    else
        cache.last_step_accepted = false
    end

    return cache.last_step_accepted, cache.u_cache, cache.fu_cache
end

# Don't Pollute the namespace
"""
    RadiusUpdateSchemes

`RadiusUpdateSchemes` is provides different types of radius update schemes implemented in
the Trust Region method. These schemes specify how the radius of the so-called trust region
is updated after each iteration of the algorithm. The specific role and caveats associated
with each scheme are provided below.

## Using `RadiusUpdateSchemes`

Simply put the desired scheme as follows:
`sol = solve(prob, alg = TrustRegion(radius_update_scheme = RadiusUpdateSchemes.Hei))`.
"""
module RadiusUpdateSchemes

using SumTypes

@sum_type RadiusUpdateScheme begin
    Simple
    NLsolve
    NocedalWright
    Hei
    Yuan
    Bastin
    Fan
end

const T = RadiusUpdateScheme

@doc """
    RadiusUpdateSchemes.Simple

The simple or conventional radius update scheme. This scheme is chosen by default and
follows the conventional approach to update the trust region radius, i.e. if the trial
step is accepted it increases the radius by a fixed factor (bounded by a maximum radius)
and if the trial step is rejected, it shrinks the radius by a fixed factor.
""" Simple

@doc """
    RadiusUpdateSchemes.NLsolve

The same updating scheme as in NLsolve's (https://github.com/JuliaNLSolvers/NLsolve.jl)
trust region dogleg implementation.
""" NLsolve

@doc """
    RadiusUpdateSchemes.NocedalWright

Trust region updating scheme as in Nocedal and Wright [see Alg 11.5, page 291].
""" NocedalWright

@doc """
    RadiusUpdateSchemes.Hei

This scheme is proposed by Hei, L. [1]. The trust region radius depends on the size
(norm) of the current step size. The hypothesis is to let the radius converge to zero as
the iterations progress, which is more reliable and robust for ill-conditioned as well
as degenerate problems.

### References

[1] Hei, Long. "A self-adaptive trust region algorithm." Journal of Computational
Mathematics (2003): 229-236.
""" Hei

@doc """
    RadiusUpdateSchemes.Yuan

This scheme is proposed by Yuan, Y [1]. Similar to Hei's scheme, the trust region is
updated in a way so that it converges to zero, however here, the radius depends on the
size (norm) of the current gradient of the objective (merit) function. The hypothesis is
that the step size is bounded by the gradient size, so it makes sense to let the radius
depend on the gradient.

### References

[1] Fan, Jinyan, Jianyu Pan, and Hongyan Song. "A retrospective trust region algorithm
with trust region converging to zero." Journal of Computational Mathematics 34.4 (2016):
421-436.
""" Yuan

@doc """
    RadiusUpdateSchemes.Bastin

This scheme is proposed by Bastin, et al. [1]. The scheme is called a retrospective
update scheme as it uses the model function at the current iteration to compute the
ratio of the actual reduction and the predicted reduction in the previous trial step,
and use this ratio to update the trust region radius. The hypothesis is to exploit the
information made available during the optimization process in order to vary the accuracy
of the objective function computation.

### References

[1] Bastin, Fabian, et al. "A retrospective trust-region method for unconstrained
optimization." Mathematical programming 123 (2010): 395-418.
""" Bastin

@doc """
    RadiusUpdateSchemes.Fan

This scheme is proposed by Fan, J. [1]. It is very much similar to Hei's and Yuan's
schemes as it lets the trust region radius depend on the current size (norm) of the
objective (merit) function itself. These new update schemes are known to improve local
convergence.

### References

[1] Fan, Jinyan. "Convergence rate of the trust region method for nonlinear equations
under local error bound condition." Computational Optimization and Applications 34.2
(2006): 215-227.
""" Fan

end

"""
    GenericTrustRegionScheme(; method = RadiusUpdateSchemes.Simple,
        max_trust_radius = nothing, initial_trust_radius = nothing,
        step_threshold = nothing, shrink_threshold = nothing, expand_threshold = nothing,
        shrink_factor = nothing, expand_factor = nothing, forward_ad = nothing,
        reverse_ad = nothing)

Trust Region Method that updates and stores the current trust region radius in
`trust_region`. For any of the keyword arguments, if the value is `nothing`, then we use
the value used in the respective paper.

### Keyword Arguments

  - `radius_update_scheme`: the choice of radius update scheme to be used. Defaults to
    `RadiusUpdateSchemes.Simple` which follows the conventional approach. Other available
    schemes are documented in [`RadiusUpdateSchemes`](@ref),. These schemes have the trust
    region radius converging to zero that is seen to improve convergence. For more details,
    see [1].
  - `max_trust_radius`: the maximal trust region radius. Defaults to
    `max(norm(fu), maximum(u) - minimum(u))`, except for `RadiusUpdateSchemes.NLsolve`
    where it defaults to `Inf`.
  - `initial_trust_radius`: the initial trust region radius. Defaults to
    `max_trust_radius / 11`, except for `RadiusUpdateSchemes.NLsolve` where it defaults
    to `u0_norm > 0 ? u0_norm : 1`.
  - `step_threshold`: the threshold for taking a step. In every iteration, the threshold is
    compared with a value `r`, which is the actual reduction in the objective function
    divided by the predicted reduction. If `step_threshold > r` the model is not a good
    approximation, and the step is rejected. Defaults to `nothing`. For more details, see
    [2].
  - `shrink_threshold`: the threshold for shrinking the trust region radius. In every
    iteration, the threshold is compared with a value `r` which is the actual reduction in
    the objective function divided by the predicted reduction. If `shrink_threshold > r` the
    trust region radius is shrunk by `shrink_factor`. Defaults to `nothing`. For more
    details, see [2].
  - `expand_threshold`: the threshold for expanding the trust region radius. If a step is
    taken, i.e `step_threshold < r` (with `r` defined in `shrink_threshold`), a check is
    also made to see if `expand_threshold < r`. If that is true, the trust region radius is
    expanded by `expand_factor`. Defaults to `nothing`.
  - `shrink_factor`: the factor to shrink the trust region radius with if
    `shrink_threshold > r` (with `r` defined in `shrink_threshold`). Defaults to `0.25`.
  - `expand_factor`: the factor to expand the trust region radius with if
    `expand_threshold < r` (with `r` defined in `shrink_threshold`). Defaults to `2.0`.

### References

[1] Yuan, Ya-xiang. "Recent advances in trust region algorithms." Mathematical Programming
151 (2015): 249-281.
[2] Rahpeymaii, Farzad. "An efficient line search trust-region for systems of nonlinear
equations." Mathematical Sciences 14.3 (2020): 257-268.
"""
@kwdef @concrete struct GenericTrustRegionScheme
    method = RadiusUpdateSchemes.Simple
    step_threshold = nothing
    shrink_threshold = nothing
    shrink_factor = nothing
    expand_factor = nothing
    expand_threshold = nothing
    max_trust_radius = nothing
    initial_trust_radius = nothing
    forward_ad = nothing
    reverse_ad = nothing
end

function Base.show(io::IO, alg::GenericTrustRegionScheme)
    print(io, "GenericTrustRegionScheme(method = $(alg.method))")
end

@concrete mutable struct GenericTrustRegionSchemeCache <: AbstractTrustRegionMethodCache
    method
    f
    p
    max_trust_radius
    initial_trust_radius
    trust_region
    step_threshold
    shrink_threshold
    expand_threshold
    shrink_factor
    expand_factor
    p1
    p2
    p3
    p4
    ϵ
    ρ
    vjp_operator
    jvp_operator
    Jᵀfu_cache
    Jδu_cache
    δu_cache
    internalnorm
    u_cache
    fu_cache
    last_step_accepted::Bool
    shrink_counter::Int
    nf::Int
    alg
end

function reinit_cache!(cache::GenericTrustRegionSchemeCache, args...; u0 = nothing,
        p = cache.p, kwargs...)
    T = eltype(cache.u_cache)
    cache.p = p
    if u0 !== nothing
        u0_norm = cache.internalnorm(u0)
        cache.trust_region = __initial_trust_radius(cache.alg.initial_trust_radius, T,
            cache.alg.method, cache.max_trust_radius, u0_norm)  # FIXME: scheme specific
    end
    cache.last_step_accepted = false
    cache.shrink_counter = 0
    cache.nf = 0
end

# Defaults
for func in (:__max_trust_radius, :__initial_trust_radius, :__step_threshold,
    :__shrink_threshold, :__shrink_factor, :__expand_threshold, :__expand_factor)
    @eval begin
        @inline function $(func)(val, ::Type{T}, args...) where {T}
            val_T = T(val)
            iszero(val_T) && return $(func)(nothing, T, args...)
            return val_T
        end
    end
end

@inline function __max_trust_radius(::Nothing, ::Type{T}, method, u, fu_norm) where {T}
    return @cases method begin
        Simple => begin
            u_min, u_max = extrema(u)
            max(T(fu_norm), u_max - u_min)
        end
        NocedalWright => begin
            u_min, u_max = extrema(u)
            max(T(fu_norm), u_max - u_min)
        end
        _ => T(Inf)
    end
end

@inline function __initial_trust_radius(::Nothing, ::Type{T}, method, max_tr,
        u0_norm) where {T}
    return @cases method begin
        NLsolve => T(ifelse(u0_norm > 0, u0_norm, 1))
        Hei => T(1)
        Bastin => T(1)
        _ => T(max_tr / 11)
    end
end

@inline function __step_threshold(::Nothing, ::Type{T}, method) where {T}
    return @cases method begin
        Hei => T(0)
        Yuan => T(1 // 1000)
        Bastin => T(1 // 20)
        _ => T(1 // 10000)
    end
end

@inline function __shrink_threshold(::Nothing, ::Type{T}, method) where {T}
    return @cases method begin
        NLsolve => T(1 // 20)
        Hei => T(0)
        Bastin => T(1 // 20)
        _ => T(1 // 4)
    end
end

@inline function __expand_threshold(::Nothing, ::Type{T}, method) where {T}
    return @cases method begin
        NLsolve => T(9 // 10)
        Hei => T(0)
        Bastin => T(9 // 10)
        _ => T(3 // 4)
    end
end

@inline function __shrink_factor(::Nothing, ::Type{T}, method) where {T}
    return @cases method begin
        NLsolve => T(1 // 2)
        Hei => T(0)
        Bastin => T(1 // 20)
        _ => T(1 // 4)
    end
end

@inline __expand_factor(::Nothing, ::Type{T}, method) where {T} = T(2)

function SciMLBase.init(prob::AbstractNonlinearProblem, alg::GenericTrustRegionScheme,
        f::F, fu, u, p, args...; internalnorm::IF = DEFAULT_NORM, kwargs...) where {F, IF}
    T = promote_type(eltype(u), eltype(fu))
    u0_norm = internalnorm(u)
    fu_norm = internalnorm(fu)

    # Common Setup
    max_trust_radius = __max_trust_radius(alg.max_trust_radius, T, alg.method, u, fu_norm)
    initial_trust_radius = __initial_trust_radius(alg.initial_trust_radius, T, alg.method,
        max_trust_radius, u0_norm)
    step_threshold = __step_threshold(alg.step_threshold, T, alg.method)
    shrink_threshold = __shrink_threshold(alg.shrink_threshold, T, alg.method)
    expand_threshold = __expand_threshold(alg.expand_threshold, T, alg.method)
    shrink_factor = __shrink_factor(alg.shrink_factor, T, alg.method)
    expand_factor = __expand_factor(alg.expand_factor, T, alg.method)

    # Scheme Specific Setup
    p1, p2, p3, p4 = ntuple(_ -> T(0), 4)
    ϵ, vjp_operator, jvp_operator, δu_cache = T(1e-8), nothing, nothing, nothing

    @cases alg.method begin
        NLsolve => (p1 = T(1 // 2))
        Hei => begin
            p1, p2, p3, p4 = T(5), T(1 // 10), T(15 // 100), T(15 // 100)
        end
        Yuan => begin
            p1, p2, p3 = T(2), T(1 // 6), T(6)
            vjp_operator = VecJacOperator(prob, fu, u; autodiff = alg.reverse_ad)
        end
        Fan => begin
            p1, p2, p3, p4 = T(1 // 10), T(1 // 4), T(12), T(1e18)
            initial_trust_radius = T(p1 * fu_norm^0.99)
        end
        Bastin => begin
            p1, p2 = T(5 // 2), T(1 // 4)
            vjp_operator = VecJacOperator(prob, fu, u; autodiff = alg.reverse_ad)
            jvp_operator = JacVecOperator(prob, fu, u; autodiff = alg.forward_ad)
            @bb δu_cache = similar(u)
        end
        _ => ()
    end

    Jᵀfu_cache = nothing
    @cases alg.method begin
        Yuan => begin
            Jᵀfu_cache = StatefulJacobianOperator(vjp_operator, u, prob.p) * _vec(fu)
            initial_trust_radius = T(p1 * internalnorm(Jᵀfu_cache))
        end
        _ => begin
            if u isa Number
                Jᵀfu_cache = u
            else
                @bb Jᵀfu_cache = similar(u)
            end
        end
    end

    @bb u_cache = similar(u)
    @bb fu_cache = similar(fu)
    @bb Jδu_cache = similar(fu)

    return GenericTrustRegionSchemeCache(alg.method, f, p, max_trust_radius,
        initial_trust_radius, initial_trust_radius, step_threshold, shrink_threshold,
        expand_threshold, shrink_factor, expand_factor, p1, p2, p3, p4, ϵ, T(0),
        vjp_operator, jvp_operator, Jᵀfu_cache, Jδu_cache, δu_cache, internalnorm,
        u_cache, fu_cache, false, 0, 0, alg)
end

function SciMLBase.solve!(cache::GenericTrustRegionSchemeCache, J, fu, u, δu, damping_stats)
    @bb @. cache.u_cache = u + δu
    cache.fu_cache = evaluate_f!!(cache.f, cache.fu_cache, cache.u_cache, cache.p)
    cache.nf += 1

    @bb cache.Jδu_cache = J × vec(δu)
    @bb cache.Jᵀfu_cache = transpose(J) × vec(fu)
    num = (cache.internalnorm(cache.fu_cache)^2 - cache.internalnorm(fu)^2) / 2
    denom = __dot(δu, cache.Jᵀfu_cache) + __dot(cache.Jδu_cache, cache.Jδu_cache) / 2
    cache.ρ = num / denom

    if cache.ρ > cache.step_threshold
        cache.last_step_accepted = true
    else
        cache.last_step_accepted = false
    end

    @cases cache.method begin
        Simple => begin
            if cache.ρ < cache.shrink_threshold
                cache.trust_region *= cache.shrink_factor
                cache.shrink_counter += 1
            else
                cache.shrink_counter = 0
                if cache.ρ > cache.step_threshold && cache.ρ > cache.expand_threshold
                    cache.trust_region = cache.expand_factor * cache.trust_region
                end
            end
        end
        NLsolve => begin
            if cache.ρ < cache.shrink_threshold
                cache.trust_region *= cache.shrink_factor
                cache.shrink_counter += 1
            else
                cache.shrink_counter = 0
                if cache.ρ ≥ cache.expand_threshold
                    cache.trust_region = cache.expand_factor * cache.internalnorm(δu)
                elseif cache.ρ ≥ cache.p1
                    cache.trust_region = max(cache.trust_region,
                        cache.expand_factor * cache.internalnorm(δu))
                end
            end
        end
        NocedalWright => begin
            if cache.ρ < cache.shrink_threshold
                cache.trust_region = cache.shrink_factor * cache.internalnorm(δu)
                cache.shrink_counter += 1
            else
                cache.shrink_counter = 0
                if cache.ρ > cache.expand_threshold &&
                   abs(cache.internalnorm(δu) - cache.trust_region) <
                   1e-6 * cache.trust_region
                    cache.trust_region = cache.expand_factor * cache.trust_region
                end
            end
        end
        Hei => begin
            tr_new = __rfunc(cache.ρ, cache.shrink_threshold, cache.p1, cache.p3, cache.p4,
                cache.p2) * cache.internalnorm(δu)
            if tr_new < cache.trust_region
                cache.shrink_counter += 1
            else
                cache.shrink_counter = 0
            end
            cache.trust_region = tr_new
        end
        Yuan => begin
            if cache.ρ < cache.shrink_threshold
                cache.p1 = cache.p2 * cache.p1
                cache.shrink_counter += 1
            else
                if cache.ρ ≥ cache.expand_threshold &&
                   2 * cache.internalnorm(δu) > cache.trust_region
                    cache.p1 = cache.p3 * cache.p1
                end
                cache.shrink_counter = 0
            end
            operator = StatefulJacobianOperator(cache.vjp_operator, cache.u_cache, cache.p)
            @bb cache.Jᵀfu_cache = operator × vec(cache.fu_cache)
            cache.trust_region = cache.p1 * cache.internalnorm(cache.Jᵀfu_cache)
        end
        Fan => begin
            if cache.ρ < cache.shrink_threshold
                cache.p1 *= cache.p2
                cache.shrink_counter += 1
            else
                cache.shrink_counter = 0
                cache.ρ > cache.expand_threshold && (cache.p1 = min(cache.p1 * cache.p3,
                    cache.p4))
            end
            cache.trust_region = cache.p1 * (cache.internalnorm(cache.fu_cache)^0.99)
        end
        Bastin => begin
            if cache.ρ > cache.step_threshold
                jvp_op = StatefulJacobianOperator(cache.jvp_operator, cache.u_cache,
                    cache.p)
                vjp_op = StatefulJacobianOperator(cache.vjp_operator, cache.u_cache,
                    cache.p)
                @bb cache.Jδu_cache = jvp_op × vec(δu)
                @bb cache.Jᵀfu_cache = vjp_op × vec(cache.fu_cache)
                denom_1 = dot(_vec(δu), cache.Jᵀfu_cache)
                @bb cache.Jᵀfu_cache = vjp_op × vec(cache.Jδu_cache)
                denom_2 = dot(_vec(δu), cache.Jᵀfu_cache)
                denom = denom_1 + denom_2 / 2
                ρ = num / denom
                if ρ ≥ cache.expand_threshold
                    cache.trust_region = cache.p1 * cache.internalnorm(δu)
                end
                cache.shrink_counter = 0
            else
                cache.trust_region *= cache.p2
                cache.shrink_counter += 1
            end
        end
    end

    cache.trust_region = min(cache.trust_region, cache.max_trust_radius)

    return cache.last_step_accepted, cache.u_cache, cache.fu_cache
end

# R-function for adaptive trust region method
function __rfunc(r::R, c2::R, M::R, γ1::R, γ2::R, β::R) where {R <: Real}
    return ifelse(r ≥ c2,
        (2 * (M - 1 - γ2) * atan(r - c2) + (1 + γ2)) / R(π),
        (1 - γ1 - β) * (exp(r - c2) + β / (1 - γ1 - β)))
end
