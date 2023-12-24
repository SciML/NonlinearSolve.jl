function curve_fit end

function curve_fit(f::F, xdata, ydata, p₀, args...; kwargs...) where {F <: Function}
    return curve_fit(NonlinearFunction(f), xdata, ydata, p₀, solver; kwargs...)
end

function curve_fit(model::NonlinearFunction{iip, specialize}, xdata, ydata, p₀,
        args...; kwargs...) where {iip, specialize}
    __check_data_health(xdata, ydata)

    f = if iip
        @closure (du, u, p) -> begin
            model.f(du, u, p)
            @. du .-= ydata
        end
    else
        @closure (u, p) -> model.f(u, p) .- ydata
    end

    nlf = __replace_nonlinearfunction(model, f)
    nlprob = NonlinearLeastSquaresProblem{iip}(nlf, p₀, xdata)
    return solve(nlprob, args...; kwargs...)
end

@inline function __check_data_health(xdata, ydata)
    if any(ismissing, xdata) || any(ismissing, ydata)
        error("Data contains `missing` values and a fit cannot be performed")
    end
    if any(isinf, xdata) || any(isinf, ydata) || any(isnan, xdata) || any(isnan, ydata)
        error("Data contains `Inf` or `NaN` values and a fit cannot be performed")
    end
end
