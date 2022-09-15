"""
    @add_kwonly function_definition

Define keyword-only version of the `function_definition`.

    @add_kwonly function f(x; y=1)
        ...
    end

expands to:

    function f(x; y=1)
        ...
    end
    function f(; x = error("No argument x"), y=1)
        ...
    end
"""
macro add_kwonly(ex)
    esc(add_kwonly(ex))
end

add_kwonly(ex::Expr) = add_kwonly(Val{ex.head}, ex)

function add_kwonly(::Type{<:Val}, ex)
    error("add_only does not work with expression $(ex.head)")
end

function add_kwonly(::Union{Type{Val{:function}},
                            Type{Val{:(=)}}}, ex::Expr)
    body = ex.args[2:end]  # function body
    default_call = ex.args[1]  # e.g., :(f(a, b=2; c=3))
    kwonly_call = add_kwonly(default_call)
    if kwonly_call === nothing
        return ex
    end

    return quote
        begin
            $ex
            $(Expr(ex.head, kwonly_call, body...))
        end
    end
end

function add_kwonly(::Type{Val{:where}}, ex::Expr)
    default_call = ex.args[1]
    rest = ex.args[2:end]
    kwonly_call = add_kwonly(default_call)
    if kwonly_call === nothing
        return nothing
    end
    return Expr(:where, kwonly_call, rest...)
end

function add_kwonly(::Type{Val{:call}}, default_call::Expr)
    # default_call is, e.g., :(f(a, b=2; c=3))
    funcname = default_call.args[1]  # e.g., :f
    required = []  # required positional arguments; e.g., [:a]
    optional = []  # optional positional arguments; e.g., [:(b=2)]
    default_kwargs = []
    for arg in default_call.args[2:end]
        if isa(arg, Symbol)
            push!(required, arg)
        elseif arg.head == :(::)
            push!(required, arg)
        elseif arg.head == :kw
            push!(optional, arg)
        elseif arg.head == :parameters
            @assert default_kwargs == []  # can I have :parameters twice?
            default_kwargs = arg.args
        else
            error("Not expecting to see: $arg")
        end
    end
    if isempty(required) && isempty(optional)
        # If the function is already keyword-only, do nothing:
        return nothing
    end
    if isempty(required)
        # It's not clear what should be done.  Let's not support it at
        # the moment:
        error("At least one positional mandatory argument is required.")
    end

    kwonly_kwargs = Expr(:parameters,
                         [Expr(:kw, pa, :(error($("No argument $pa"))))
                          for pa in required]..., optional..., default_kwargs...)
    kwonly_call = Expr(:call, funcname, kwonly_kwargs)
    # e.g., :(f(; a=error(...), b=error(...), c=1, d=2))

    return kwonly_call
end

function num_types_in_tuple(sig)
    length(sig.parameters)
end

function num_types_in_tuple(sig::UnionAll)
    length(Base.unwrap_unionall(sig).parameters)
end

@inline UNITLESS_ABS2(x) = real(abs2(x))
@inline DEFAULT_NORM(u::Union{AbstractFloat, Complex}) = @fastmath abs(u)
@inline function DEFAULT_NORM(u::Array{T}) where {T <: Union{AbstractFloat, Complex}}
    sqrt(real(sum(abs2, u)) / length(u))
end
@inline function DEFAULT_NORM(u::StaticArray{T}) where {T <: Union{AbstractFloat, Complex}}
    sqrt(real(sum(abs2, u)) / length(u))
end
@inline function DEFAULT_NORM(u::RecursiveArrayTools.AbstractVectorOfArray)
    sum(sqrt(real(sum(UNITLESS_ABS2, _u)) / length(_u)) for _u in u.u)
end
@inline DEFAULT_NORM(u::AbstractArray) = sqrt(real(sum(UNITLESS_ABS2, u)) / length(u))
@inline DEFAULT_NORM(u) = norm(u)

"""
  prevfloat_tdir(x, x0, x1)

Move `x` one floating point towards x0.
"""
function prevfloat_tdir(x, x0, x1)
    x1 > x0 ? prevfloat(x) : nextfloat(x)
end

function nextfloat_tdir(x, x0, x1)
    x1 > x0 ? nextfloat(x) : prevfloat(x)
end

function max_tdir(a, b, x0, x1)
    x1 > x0 ? max(a, b) : min(a, b)
end

alg_autodiff(alg::AbstractNewtonAlgorithm{CS, AD}) where {CS, AD} = AD
alg_autodiff(alg) = false

"""
  value_derivative(f, x)

Compute `f(x), d/dx f(x)` in the most efficient way.
"""
function value_derivative(f::F, x::R) where {F, R}
    T = typeof(ForwardDiff.Tag(f, R))
    out = f(ForwardDiff.Dual{T}(x, one(x)))
    ForwardDiff.value(out), ForwardDiff.extract_derivative(T, out)
end

# Todo: improve this dispatch
value_derivative(f::F, x::SVector) where {F} = f(x), ForwardDiff.jacobian(f, x)

value(x) = x
value(x::Dual) = ForwardDiff.value(x)
value(x::AbstractArray{<:Dual}) = map(ForwardDiff.value, x)

_unwrap_val(::Val{B}) where {B} = B
_unwrap_val(B) = B

_vec(v) = vec(v)
_vec(v::Number) = v
_vec(v::AbstractVector) = v