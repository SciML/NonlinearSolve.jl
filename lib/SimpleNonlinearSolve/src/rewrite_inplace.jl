# Take a inplace code and rewrite it to be maybe-inplace
# I will take this code out into a separate package because this is useful even in
# NonlinearSolve.jl
function __bangbang(M, expr; depth = 1)
    new_expr = nothing
    if expr.head == :call
        @assert length(expr.args)≥2 "Expected a function call with atleast 1 argument. \
                                     Got `$(expr)`."
        f, a, args... = expr.args
        g = get(OP_MAPPING, f, nothing)
        if f == :copy && length(args) == 0
            # Special case for copy with single argument
            new_expr = :($(g)($(setindex_trait)($(a)), $(a)))
        elseif g !== nothing
            new_expr = :($(a) = $(g)($(setindex_trait)($(a)), $(a), $(args...)))
        end
    elseif expr.head == :(=)
        a, rhs_expr = expr.args
        if rhs_expr.head == :call
            f, b, args... = rhs_expr.args
            g = get(OP_MAPPING, f, nothing)
            if g !== nothing
                new_expr = :($(a) = $(g)($(setindex_trait)($(b)), $(b), $(args...)))
            elseif f == :×
                @debug "Custom operator `×` detected in `$(expr)`."
                c, args... = args
                @assert length(args)==0 "Expected `×` to have only 2 arguments. \
                                        Got `$(expr)`."
                is_b_vec = b isa Expr && b.head == :call && b.args[1] == :vec
                is_c_vec = c isa Expr && c.head == :call && c.args[1] == :vec
                a_sym = gensym("a")
                if is_b_vec
                    if is_c_vec
                        error("2 `vec`s detected with `×` in `$(expr)`. Use `dot` instead.")
                    else
                        new_expr = quote
                            if $(setindex_trait)($(a)) === CanSetindex()
                                $(a_sym) = $(_vec)($a)
                                mul!($(a_sym), $(_vec)($(b.args[2])), $(c))
                                $(a) = $(_restructure)($a, $(a_sym))
                            else
                                $(a) = $(_restructure)($a, $(_vec)($(b.args[2])) * $(c))
                            end
                        end
                    end
                else
                    if is_c_vec
                        new_expr = quote
                            if $(setindex_trait)($(a)) === CanSetindex()
                                $(a_sym) = $(_vec)($a)
                                mul!($(a), $(b), $(_vec)($(c.args[2])))
                                $(a) = $(_restructure)($a, $(a_sym))
                            else
                                $(a) = $(_restructure)($a, $(b) * $(_vec)($(c.args[2])))
                            end
                        end
                    else
                        new_expr = quote
                            if $(setindex_trait)($(a)) === CanSetindex()
                                mul!($(a), $(b), $(c))
                            else
                                $(a) = $(b) * $(c)
                            end
                        end
                    end
                end
            end
        end
    elseif expr.head == :(.=)
        a, rhs_expr = expr.args
        if rhs_expr isa Expr && rhs_expr.head == :(.)
            f, arg_expr = rhs_expr.args
            # f_broadcast = :(Base.Broadcast.BroadcastFunction($(f)))
            new_expr = quote
                if $(setindex_trait)($(a)) === CanSetindex()
                    broadcast!($(f), $(a), $(arg_expr)...)
                else
                    $(a) = broadcast($(f), $(arg_expr)...)
                end
            end
        end
    elseif expr.head == :macrocall
        # For @__dot__ there is a easier alternative
        if expr.args[1] == Symbol("@__dot__")
            main_expr = last(expr.args)
            if main_expr isa Expr && main_expr.head == :(=)
                a, rhs_expr = main_expr.args
                new_expr = quote
                    if $(setindex_trait)($(a)) === CanSetindex()
                        @. $(main_expr)
                    else
                        $(a) = @. $(rhs_expr)
                    end
                end
            end
        end
        if new_expr === nothing
            new_expr = __bangbang(M, Base.macroexpand(M, expr; recursive = true);
                depth = depth + 1)
        end
    else
        f = expr.head # Things like :.-=, etc.
        a, args... = expr.args
        g = get(OP_MAPPING, f, nothing)
        if g !== nothing
            new_expr = :($(a) = $(g)($(setindex_trait)($(a)), $(a), $(args...)))
        end
    end
    if new_expr !== nothing
        if depth == 1
            @debug "Replacing `$(expr)` with `$(new_expr)`"
            return esc(new_expr)
        else
            return new_expr
        end
    end
    error("`$(expr)` cannot be handled. Check the documentation for allowed expressions.")
end

macro bangbang(expr)
    return __bangbang(__module__, expr)
end

# `bb` is the short form of bang-bang
macro bb(expr)
    return __bangbang(__module__, expr)
end

# Is Mutable or Not?
abstract type AbstractMaybeSetindex end
struct CannotSetindex <: AbstractMaybeSetindex end
struct CanSetindex <: AbstractMaybeSetindex end

# Common types should overload this via extensions, else it butchers type-inference
setindex_trait(::Union{Number, SArray}) = CannotSetindex()
setindex_trait(::Union{MArray, Array}) = CanSetindex()
setindex_trait(A) = ifelse(ArrayInterface.can_setindex(A), CanSetindex(), CannotSetindex())

# Operations
const OP_MAPPING = Dict{Symbol, Symbol}(:copyto! => :__copyto!!,
    :.-= => :__sub!!,
    :.+= => :__add!!,
    :.*= => :__mul!!,
    :./= => :__div!!,
    :copy => :__copy)

@inline __copyto!!(::CannotSetindex, x, y) = y
@inline __copyto!!(::CanSetindex, x, y) = (copyto!(x, y); x)

@inline __broadcast!!(::CannotSetindex, op, x, args...) = broadcast(op, args...)
@inline __broadcast!!(::CanSetindex, op, x, args...) = (broadcast!(op, x, args...); x)

@inline __sub!!(S, x, args...) = __broadcast!!(S, -, x, x, args...)
@inline __add!!(S, x, args...) = __broadcast!!(S, +, x, x, args...)
@inline __mul!!(S, x, args...) = __broadcast!!(S, *, x, x, args...)
@inline __div!!(S, x, args...) = __broadcast!!(S, /, x, x, args...)

@inline __copy(::CannotSetindex, x) = x
@inline __copy(::CanSetindex, x) = copy(x)
@inline __copy(::CannotSetindex, x, y) = y
@inline __copy(::CanSetindex, x, y) = copy(y)
