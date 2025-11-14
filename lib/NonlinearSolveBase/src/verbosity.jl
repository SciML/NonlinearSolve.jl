"""
    NonlinearVerbosity <: AbstractVerbositySpecifier

Verbosity configuration for NonlinearSolve.jl solvers, providing fine-grained control over
diagnostic messages, warnings, and errors during nonlinear system solution.

# Fields

## Error Control Group
- `non_enclosing_interval`: Messages when interval doesn't enclose root (bracketing methods)
- `alias_u0_immutable`: Messages when aliasing u0 with immutable array
- `linsolve_failed_noncurrent`: Messages when linear solve fails on non-current iteration
- `termination_condition`: Messages about termination conditions

## Numerical Group
- `threshold_state`: Messages about threshold state in low-rank methods

## Linear Solver Group
- `linear_verbosity`: Verbosity configuration for linear solvers

# Constructors

    NonlinearVerbosity(preset::AbstractVerbosityPreset)

Create a `NonlinearVerbosity` using a preset configuration:
- `SciMLLogging.None()`: All messages disabled
- `SciMLLogging.Minimal()`: Only critical errors and fatal issues
- `SciMLLogging.Standard()`: Balanced verbosity (default)
- `SciMLLogging.Detailed()`: Comprehensive debugging information
- `SciMLLogging.All()`: Maximum verbosity

    NonlinearVerbosity(; error_control=nothing, performance=nothing, numerical=nothing, linear_verbosity=nothing, kwargs...)

Create a `NonlinearVerbosity` with group-level or individual field control.

# Examples

```julia
# Use a preset
verbose = NonlinearVerbosity(SciMLLogging.Standard())

# Set entire groups
verbose = NonlinearVerbosity(
    error_control = SciMLLogging.WarnLevel(),
    numerical = SciMLLogging.InfoLevel()
)

# Set individual fields
verbose = NonlinearVerbosity(
    alias_u0_immutable = SciMLLogging.WarnLevel(),
    threshold_state = SciMLLogging.InfoLevel()
)

# Mix group and individual settings
verbose = NonlinearVerbosity(
    numerical = SciMLLogging.InfoLevel(),  # Set all numerical to InfoLevel
    threshold_state = SciMLLogging.ErrorLevel()  # Override specific field
)
```
"""
@concrete struct NonlinearVerbosity <: AbstractVerbositySpecifier
    # Linear verbosity
    linear_verbosity
    # Error control
    non_enclosing_interval
    alias_u0_immutable
    linsolve_failed_noncurrent
    termination_condition
    # Numerical
    threshold_state
end

# Group classifications
const error_control_options = (
    :non_enclosing_interval, :alias_u0_immutable, :linsolve_failed_noncurrent,
    :termination_condition
)
const performance_options = ()
const numerical_options = (:threshold_state,)

function option_group(option::Symbol)
    if option in error_control_options
        return :error_control
    elseif option in performance_options
        return :performance
    elseif option in numerical_options
        return :numerical
    else
        error("Unknown verbosity option: $option")
    end
end

# Get all options in a group
function group_options(verbosity::NonlinearVerbosity, group::Symbol)
    if group === :error_control
        return NamedTuple{error_control_options}(getproperty(verbosity, opt)
                                                 for opt in error_control_options)
    elseif group === :performance
        return NamedTuple{performance_options}(getproperty(verbosity, opt)
                                               for opt in performance_options)
    elseif group === :numerical
        return NamedTuple{numerical_options}(getproperty(verbosity, opt)
                                             for opt in numerical_options)
    else
        error("Unknown group: $group")
    end
end

function NonlinearVerbosity(;
        error_control = nothing, performance = nothing, numerical = nothing,
        linear_verbosity = nothing, kwargs...)
    # Validate group arguments
    if error_control !== nothing && !(error_control isa AbstractMessageLevel)
        throw(ArgumentError("error_control must be a SciMLLogging.AbstractMessageLevel, got $(typeof(error_control))"))
    end
    if performance !== nothing && !(performance isa AbstractMessageLevel)
        throw(ArgumentError("performance must be a SciMLLogging.AbstractMessageLevel, got $(typeof(performance))"))
    end
    if numerical !== nothing && !(numerical isa AbstractMessageLevel)
        throw(ArgumentError("numerical must be a SciMLLogging.AbstractMessageLevel, got $(typeof(numerical))"))
    end

    # Validate individual kwargs
    for (key, value) in kwargs
        if !(key in error_control_options || key in performance_options ||
             key in numerical_options)
            throw(ArgumentError("Unknown verbosity option: $key. Valid options are: $(tuple(error_control_options..., performance_options..., numerical_options...))"))
        end
        if !(value isa AbstractMessageLevel)
            throw(ArgumentError("$key must be a SciMLLogging.AbstractMessageLevel, got $(typeof(value))"))
        end
    end

    # Build arguments using NamedTuple for type stability
    default_args = (
        linear_verbosity = linear_verbosity === nothing ? Minimal() : linear_verbosity,
        non_enclosing_interval = WarnLevel(),
        alias_u0_immutable = WarnLevel(),
        linsolve_failed_noncurrent = WarnLevel(),
        termination_condition = WarnLevel(),
        threshold_state = WarnLevel()
    )

    # Apply group-level settings
    final_args = if error_control !== nothing || performance !== nothing ||
                    numerical !== nothing
        NamedTuple{keys(default_args)}(
            _resolve_arg_value(
                key, default_args[key], error_control, performance, numerical)
        for key in keys(default_args)
        )
    else
        default_args
    end

    # Apply individual overrides
    if !isempty(kwargs)
        final_args = merge(final_args, NamedTuple(kwargs))
    end

    NonlinearVerbosity(values(final_args)...)
end

# Constructor for verbosity presets following the hierarchical levels:
# None < Minimal < Standard < Detailed < All
# Each level includes all messages from levels below it plus additional ones
function NonlinearVerbosity(verbose::AbstractVerbosityPreset)
    if verbose isa Minimal
        # Minimal: Only fatal errors and critical warnings
        NonlinearVerbosity(
            linear_verbosity = Minimal(),
            non_enclosing_interval = WarnLevel(),
            alias_u0_immutable = Silent(),
            linsolve_failed_noncurrent = WarnLevel(),
            termination_condition = Silent(),
            threshold_state = Silent()
        )
    elseif verbose isa Standard
        # Standard: Everything from Minimal + non-fatal warnings
        NonlinearVerbosity()
    elseif verbose isa Detailed
        # Detailed: Everything from Standard + debugging/solver behavior
        NonlinearVerbosity(
            linear_verbosity = Detailed(),
            non_enclosing_interval = WarnLevel(),
            alias_u0_immutable = WarnLevel(),
            linsolve_failed_noncurrent = WarnLevel(),
            termination_condition = WarnLevel(),
            threshold_state = WarnLevel()
        )
    elseif verbose isa All
        # All: Maximum verbosity - every possible logging message at InfoLevel
        NonlinearVerbosity(
            linear_verbosity = Detailed(),
            non_enclosing_interval = WarnLevel(),
            alias_u0_immutable = WarnLevel(),
            linsolve_failed_noncurrent = WarnLevel(),
            termination_condition = WarnLevel(),
            threshold_state = InfoLevel()
        )
    end
end

@inline function NonlinearVerbosity(verbose::None)
    NonlinearVerbosity(
        None(),
        Silent(),
        Silent(),
        Silent(),
        Silent(),
        Silent()
    )
end

# Helper function to resolve argument values based on group membership
@inline function _resolve_arg_value(
        key::Symbol, default_val, error_control, performance, numerical)
    if key === :linear_verbosity
        return default_val
    elseif key in error_control_options && error_control !== nothing
        return error_control
    elseif key in performance_options && performance !== nothing
        return performance
    elseif key in numerical_options && numerical !== nothing
        return numerical
    else
        return default_val
    end
end