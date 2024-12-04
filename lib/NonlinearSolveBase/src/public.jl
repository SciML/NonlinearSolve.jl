# forward declarations of public API
function UNITLESS_ABS2 end
function NAN_CHECK end
function L2_NORM end
function Linf_NORM end
function get_tolerance end

# Forward declarations of functions for forward mode AD
function nonlinearsolve_forwarddiff_solve end
function nonlinearsolve_dual_solution end
function nonlinearsolve_∂f_∂p end
function nonlinearsolve_∂f_∂u end
function nlls_generate_vjp_function end
function nodual_value end

"""
    pickchunksize(x) = pickchunksize(length(x))
    pickchunksize(x::Int)

Determine the chunk size for ForwardDiff and PolyesterForwardDiff based on the input length.
"""
function pickchunksize end

# Nonlinear Solve Termination Conditions
abstract type AbstractNonlinearTerminationMode end
abstract type AbstractSafeNonlinearTerminationMode <: AbstractNonlinearTerminationMode end
abstract type AbstractSafeBestNonlinearTerminationMode <:
              AbstractSafeNonlinearTerminationMode end

#! format: off
const TERM_DOCS = Dict(
    :Norm => doc"``\| \Delta u \| \leq reltol \times \| \Delta u + u \|`` or ``\| \Delta u \| \leq abstol``.",
    :Rel => doc"``all \left(| \Delta u | \leq reltol \times | u | \right)``.",
    :RelNorm => doc"``\| \Delta u \| \leq reltol \times \| \Delta u + u \|``.",
    :Abs => doc"``all \left( | \Delta u | \leq abstol \right)``.",
    :AbsNorm => doc"``\| \Delta u \| \leq abstol``."
)

const TERM_INTERNALNORM_DOCS = """
where `internalnorm` is the norm to use for the termination condition. Special handling is
done for `norm(_, 2)`, `norm`, `norm(_, Inf)`, and `maximum(abs, _)`."""
#! format: on

for name in (:Rel, :Abs)
    struct_name = Symbol(name, :TerminationMode)
    doctring = TERM_DOCS[name]

    @eval begin
        """
            $($struct_name) <: AbstractNonlinearTerminationMode

        Terminates if $($doctring).

        ``\\Delta u`` denotes the increment computed by the nonlinear solver and ``u`` denotes the solution.
        """
        struct $(struct_name) <: AbstractNonlinearTerminationMode end
    end
end

for name in (:Norm, :RelNorm, :AbsNorm)
    struct_name = Symbol(name, :TerminationMode)
    doctring = TERM_DOCS[name]

    @eval begin
        """
            $($struct_name) <: AbstractNonlinearTerminationMode

        Terminates if $($doctring).

        ``\\Delta u`` denotes the increment computed by the inner nonlinear solver.

        ## Constructor

            $($struct_name)(internalnorm = nothing)

        $($TERM_INTERNALNORM_DOCS).
        """
        struct $(struct_name){F} <: AbstractNonlinearTerminationMode
            internalnorm::F

            function $(struct_name)(internalnorm::F) where {F}
                norm = Utils.standardize_norm(internalnorm)
                return new{typeof(norm)}(norm)
            end
        end
    end
end

for norm_type in (:RelNorm, :AbsNorm), safety in (:Safe, :SafeBest)
    struct_name = Symbol(norm_type, safety, :TerminationMode)
    supertype_name = Symbol(:Abstract, safety, :NonlinearTerminationMode)

    doctring = safety == :Safe ?
               "Essentially [`$(norm_type)TerminationMode`](@ref) + terminate if there \
                has been no improvement for the last `patience_steps` + terminate if the \
                solution blows up (diverges)." :
               "Essentially [`$(norm_type)SafeTerminationMode`](@ref), but caches the best\
                solution found so far."

    @eval begin
        """
            $($struct_name) <: $($supertype_name)

        $($doctring)

        ## Constructor

            $($struct_name)(
                internalnorm; protective_threshold = nothing,
                patience_steps = 100, patience_objective_multiplier = 3,
                min_max_factor = 1.3, max_stalled_steps = nothing
            )

        $($TERM_INTERNALNORM_DOCS).
        """
        @concrete struct $(struct_name) <: $(supertype_name)
            internalnorm
            protective_threshold
            patience_steps::Int
            patience_objective_multiplier
            min_max_factor
            max_stalled_steps <: Union{Nothing, Int}

            function $(struct_name)(internalnorm::F; protective_threshold = nothing,
                    patience_steps = 100, patience_objective_multiplier = 3,
                    min_max_factor = 1.3, max_stalled_steps = nothing) where {F}
                norm = Utils.standardize_norm(internalnorm)
                return new{typeof(norm), typeof(protective_threshold),
                    typeof(patience_objective_multiplier),
                    typeof(min_max_factor), typeof(max_stalled_steps)}(
                    norm, protective_threshold, patience_steps,
                    patience_objective_multiplier, min_max_factor, max_stalled_steps)
            end
        end
    end
end
