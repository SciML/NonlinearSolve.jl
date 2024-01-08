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

@kwdef @concrete struct GenericTrustRegionScheme
    method = RadiusUpdateSchemes.Simple
    step_threshold
    shrink_threshold
    shrink_factor
    expand_factor
    expand_threshold
    max_trust_radius = 0 // 1
    initial_trust_radius = 0 // 1
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
    Jᵀfu_cache
    Jδu_cache
    r_predict
    internalnorm
    u_cache
    fu_cache
    last_step_accepted::Bool
    shrink_counter::Int
    nf::Int
end

function SciMLBase.init(prob::AbstractNonlinearProblem, alg::GenericTrustRegionScheme,
        f::F, fu, u, p, args...; internalnorm::IF = DEFAULT_NORM, kwargs...) where {F, IF}
    T = promote_type(eltype(u), eltype(fu))
    u0_norm = internalnorm(u)
    fu_norm = internalnorm(fu)

    max_trust_radius = T(0)
    initial_trust_radius = T(0)
    @cases alg.method begin
        NLsolve => begin
            max_trust_radius = T(Inf)
            initial_trust_radius = u0_norm > 0 ? T(u0_norm) : T(1)
        end
        _ => begin
            max_trust_radius = T(alg.max_trust_radius)
            if iszero(max_trust_radius)
                u_min, u_max = extrema(u)
                max_trust_radius = T(max(fu_norm, u_max - u_min))
            end
            initial_trust_radius = T(alg.initial_trust_radius)
            iszero(initial_trust_radius) &&
                (initial_trust_radius = T(max_trust_radius) / 11)
        end
    end

    step_threshold = T(alg.step_threshold)
    shrink_threshold = T(alg.shrink_threshold)
    expand_threshold = T(alg.expand_threshold)
    shrink_factor = T(alg.shrink_factor)
    expand_factor = T(alg.expand_factor)
    p1, p2, p3, p4 = ntuple(_ -> T(0), 4)
    ϵ = T(1e-8)
    vjp_operator = nothing
    Jᵀfu_cache = nothing

    @cases alg.method begin
        NLsolve => begin
            p1 = T(1 // 2)
        end
        Hei => begin
            step_threshold = T(0 // 1)
            shrink_threshold = T(1 // 4)
            expand_threshold = T(1 // 4)
            p1, p2, p3, p4 = T(5), T(0.1), T(0.15), T(0.15)
            initial_trust_radius = T(1)
        end
        Yuan => begin
            step_threshold = T(0.001)
            shrink_threshold = T(1 // 4)
            expand_threshold = T(1 // 4)
            p1, p2, p3 = T(2), T(1 // 6), T(6.0)
            vjp_operator = VecJacOperator(prob, fu, u; vjp_autodiff = alg.reverse_ad,
                skip_jvp = True)
            Jᵀfu_cache = vjp_operator * _vec(fu)
            initial_trust_radius = T(p1 * internalnorm(Jᵀfu_cache))
        end
        Fan => begin
            step_threshold = T(0.0001)
            shrink_threshold = T(1 // 4)
            expand_threshold = T(3 // 4)
            p1, p2, p3, p4 = T(0.1), T(1 // 4), T(12.0), T(1e18)
            initial_trust_radius = T(p1 * internalnorm(fu)^0.99)
        end
        Bastin => begin
            step_threshold = T(1 // 20)
            shrink_threshold = T(1 // 20)
            expand_threshold = T(9 // 10)
            p1, p2 = T(2.5), T(1 // 4)
            initial_trust_radius = T(1)
        end
        _ => ()
    end

    @bb u_cache = similar(u)
    @bb fu_cache = similar(fu)
    @bb Jδu_cache = similar(fu)
    @bb r_predict = similar(fu)

    return GenericTrustRegionSchemeCache(alg.method, f, p, max_trust_radius,
        initial_trust_radius, initial_trust_radius, step_threshold, shrink_threshold,
        expand_threshold, shrink_factor, expand_factor, p1, p2, p3, p4, ϵ, T(0),
        vjp_operator, Jᵀfu_cache, Jδu_cache, r_predict, internalnorm, u_cache,
        fu_cache, false, 0, 0)
end

function SciMLBase.solve!(cache::GenericTrustRegionSchemeCache, J, fu, u, δu, damping_stats)
    @bb cache.Jδu_cache = J × vec(δu)
    @bb @. cache.r_predict = fu + cache.Jδu_cache
    @bb @. cache.u_cache = u + δu
    cache.fu_cache = evaluate_f!!(cache.f, cache.fu_cache, cache.u_cache, cache.p)
    cache.nf += 1

    fu_abs2_sum = sum(abs2, fu)
    cache.ρ = (fu_abs2_sum - sum(abs2, cache.fu_cache)) /
              (fu_abs2_sum - sum(abs2, cache.r_predict))

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
                    cache.trust_region = min(cache.expand_factor * cache.trust_region,
                        cache.max_trust_radius)
                end
            end
        end
        NLsolve => begin
            if cache.ρ < 1 // 10
                cache.shrink_counter += 1
                cache.trust_region *= 1 // 2
            else
                cache.shrink_counter = 0
                if cache.ρ ≥ 9 // 10
                    cache.trust_region = 2 * cache.internalnorm(δu)
                elseif cache.ρ ≥ 1 // 2
                    cache.trust_region = max(cache.trust_region, 2 * cache.internalnorm(δu))
                end
            end
        end
        NocedalWright => begin
            if cache.ρ < 1 // 4
                cache.shrink_counter += 1
                cache.trust_region = (1 // 4) * cache.internalnorm(δu)
            else
                cache.shrink_counter = 0
                if cache.ρ > 3 // 4 &&
                   abs(cache.internalnorm(δu) - cache.trust_region) <
                   1e-6 * cache.trust_region
                    cache.trust_region = min(2 * cache.trust_region, cache.max_trust_radius)
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
                   cache.internalnorm(δu) > cache.trust_region / 2
                    cache.p1 = cache.p3 * cache.p1
                end
                cache.shrink_counter = 0
            end

            @bb cache.Jᵀfu_cache = cache.vjp_operator × vec(cache.fu)
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
            cache.trust_region = cache.p1 * (cache.internalnorm(cache.fu)^0.99)
        end
        Bastin => begin
            # TODO: retrospective step
            error("Not implemented yet")
            if cache.ρ > cache.step_threshold
                #     if retrospective_step!(cache) ≥ cache.expand_threshold
                #         cache.trust_region = max(cache.p1 * cache.internalnorm(cache.du),
                #             cache.trust_region)
                #     end
                cache.shrink_counter = 0
            else
                cache.trust_region *= cache.p2
                cache.shrink_counter += 1
            end
        end
    end

    return cache.last_step_accepted, cache.u_cache, cache.fu_cache
end

# R-function for adaptive trust region method
function __rfunc(r::R, c2::R, M::R, γ1::R, γ2::R, β::R) where {R <: Real}
    return ifelse(r ≥ c2,
        (2 * (M - 1 - γ2) * atan(r - c2) + (1 + γ2)) / R(π),
        (1 - γ1 - β) * (exp(r - c2) + β / (1 - γ1 - β)))
end

# function retrospective_step!(cache::TrustRegionCache{iip}) where {iip}
#     J = jacobian!!(cache.J_cache, cache)
#     __update_JᵀJ!(cache, J)
#     __update_Jᵀf!(cache, J)

#     num = __trust_region_loss(cache, cache.fu) - __trust_region_loss(cache, cache.fu_cache)
#     denom = dot(_vec(cache.du), _vec(cache.Jᵀf)) + __lr_mul(cache, cache.JᵀJ, cache.du) / 2
#     return num / denom
# end
