"""
    NonlinearVerbosity <: AbstractVerbositySpecifier

Verbosity configuration for NonlinearSolve.jl solvers, providing fine-grained control over
diagnostic messages, warnings, and errors during nonlinear system solution.

# Fields

## Error Control Group
- `immutable_u0`: Messages when u0 is immutable
- `non_enclosing_interval`: Messages when interval doesn't enclose root
- `non_forward_mode`: Messages when forward mode AD is not used
- `fd_ad_caution`: Messages about finite differencing cautions
- `ad_backend_incompatible`: Messages when AD backend is incompatible
- `alias_u0_immutable`: Messages when aliasing u0 with immutable array
- `linsolve_failed_noncurrent`: Messages when linear solve fails on non-current iteration
- `jacobian_free`: Messages about jacobian-free methods
- `termination_condition`: Messages about termination conditions

## Performance Group
- `colorvec_non_sparse`: Messages when color vector is used with non-sparse matrix
- `colorvec_no_prototype`: Messages when color vector has no prototype
- `sparsity_using_jac_prototype`: Messages when using jacobian prototype for sparsity
- `sparse_matrixcolorings_not_loaded`: Messages when SparseMatrixColorings not loaded

## Numerical Group
- `threshold_state`: Messages about threshold state
- `pinv_undefined`: Messages when pseudoinverse is undefined

## Linear Solver
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
    immutable_u0 = SciMLLogging.WarnLevel(),
    threshold_state = SciMLLogging.InfoLevel()
)

# Mix group and individual settings
verbose = NonlinearVerbosity(
    numerical = SciMLLogging.InfoLevel(),  # Set all numerical to InfoLevel
    pinv_undefined = SciMLLogging.ErrorLevel()  # Override specific field
)
```
"""
@concrete struct NonlinearVerbosity <: AbstractVerbositySpecifier
    # Linear verbosity
    linear_verbosity
    # Error control
    immutable_u0
    non_enclosing_interval
    non_forward_mode
    fd_ad_caution
    ad_backend_incompatible
    alias_u0_immutable
    linsolve_failed_noncurrent
    jacobian_free
    termination_condition
    # Performance
    colorvec_non_sparse
    colorvec_no_prototype
    sparsity_using_jac_prototype
    sparse_matrixcolorings_not_loaded
    # Numerical
    threshold_state
    pinv_undefined
end

# Group classifications
const error_control_options = (
    :immutable_u0, :non_enclosing_interval, :non_forward_mode, :fd_ad_caution,
    :ad_backend_incompatible, :alias_u0_immutable, :linsolve_failed_noncurrent,
    :jacobian_free, :termination_condition
)
const performance_options = (
    :colorvec_non_sparse, :colorvec_no_prototype, :sparsity_using_jac_prototype,
    :sparse_matrixcolorings_not_loaded
)
const numerical_options = (:threshold_state, :pinv_undefined)

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
        immutable_u0 = WarnLevel(),
        non_enclosing_interval = WarnLevel(),
        non_forward_mode = WarnLevel(),
        fd_ad_caution = WarnLevel(),
        ad_backend_incompatible = WarnLevel(),
        alias_u0_immutable = WarnLevel(),
        linsolve_failed_noncurrent = WarnLevel(),
        jacobian_free = WarnLevel(),
        termination_condition = WarnLevel(),
        colorvec_non_sparse = WarnLevel(),
        colorvec_no_prototype = WarnLevel(),
        sparsity_using_jac_prototype = WarnLevel(),
        sparse_matrixcolorings_not_loaded = WarnLevel(),
        threshold_state = WarnLevel(),
        pinv_undefined = WarnLevel()
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
            immutable_u0 = WarnLevel(),
            non_enclosing_interval = WarnLevel(),
            non_forward_mode = Silent(),
            fd_ad_caution = Silent(),
            ad_backend_incompatible = WarnLevel(),
            alias_u0_immutable = Silent(),
            linsolve_failed_noncurrent = WarnLevel(),
            jacobian_free = Silent(),
            termination_condition = Silent(),
            colorvec_non_sparse = Silent(),
            colorvec_no_prototype = Silent(),
            sparsity_using_jac_prototype = Silent(),
            sparse_matrixcolorings_not_loaded = Silent(),
            threshold_state = Silent(),
            pinv_undefined = ErrorLevel()
        )
    elseif verbose isa Standard
        # Standard: Everything from Minimal + non-fatal warnings
        NonlinearVerbosity()
    elseif verbose isa Detailed
        # Detailed: Everything from Standard + debugging/solver behavior
        NonlinearVerbosity(
            linear_verbosity = Minimal(),
            immutable_u0 = WarnLevel(),
            non_enclosing_interval = WarnLevel(),
            non_forward_mode = InfoLevel(),
            fd_ad_caution = WarnLevel(),
            ad_backend_incompatible = WarnLevel(),
            alias_u0_immutable = WarnLevel(),
            linsolve_failed_noncurrent = WarnLevel(),
            jacobian_free = InfoLevel(),
            termination_condition = WarnLevel(),
            colorvec_non_sparse = InfoLevel(),
            colorvec_no_prototype = InfoLevel(),
            sparsity_using_jac_prototype = InfoLevel(),
            sparse_matrixcolorings_not_loaded = InfoLevel(),
            threshold_state = WarnLevel(),
            pinv_undefined = WarnLevel()
        )
    elseif verbose isa All
        # All: Maximum verbosity - every possible logging message at InfoLevel
        NonlinearVerbosity(
            linear_verbosity = All(),
            immutable_u0 = WarnLevel(),
            non_enclosing_interval = WarnLevel(),
            non_forward_mode = InfoLevel(),
            fd_ad_caution = WarnLevel(),
            ad_backend_incompatible = WarnLevel(),
            alias_u0_immutable = WarnLevel(),
            linsolve_failed_noncurrent = WarnLevel(),
            jacobian_free = InfoLevel(),
            termination_condition = WarnLevel(),
            colorvec_non_sparse = InfoLevel(),
            colorvec_no_prototype = InfoLevel(),
            sparsity_using_jac_prototype = InfoLevel(),
            sparse_matrixcolorings_not_loaded = InfoLevel(),
            threshold_state = InfoLevel(),
            pinv_undefined = WarnLevel()
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
        Silent(),
        Silent(),
        Silent(),
        Silent(),
        Silent(),
        Silent(),
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

function Base.getproperty(verbosity::NonlinearVerbosity, name::Symbol)
    # Check if this is a group name
    if name === :error_control
        return group_options(verbosity, :error_control)
    elseif name === :performance
        return group_options(verbosity, :performance)
    elseif name === :numerical
        return group_options(verbosity, :numerical)
    else
        # Fall back to default field access
        return getfield(verbosity, name)
    end
end
