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
- `forcing`: Messages about forcing parameter in Newton-Krylov methods

## Sensitivity Group
- `sensitivity_vjp_choice`: Messages about VJP choice in sensitivity analysis (used by SciMLSensitivity.jl)

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

    NonlinearVerbosity(; error_control=nothing, numerical=nothing, sensitivity=nothing, linear_verbosity=nothing, kwargs...)

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
NonlinearVerbosity

@verbosity_specifier NonlinearVerbosity begin
    toggles = (
        :linear_verbosity, :non_enclosing_interval, :alias_u0_immutable,
        :linsolve_failed_noncurrent, :termination_condition, :threshold_state, :forcing,
        :sensitivity_vjp_choice,
    )

    presets = (
        None = (
            linear_verbosity = None(),
            non_enclosing_interval = Silent(),
            alias_u0_immutable = Silent(),
            linsolve_failed_noncurrent = Silent(),
            termination_condition = Silent(),
            threshold_state = Silent(),
            forcing = Silent(),
            sensitivity_vjp_choice = Silent(),
        ),
        Minimal = (
            linear_verbosity = None(),
            non_enclosing_interval = WarnLevel(),
            alias_u0_immutable = Silent(),
            linsolve_failed_noncurrent = WarnLevel(),
            termination_condition = Silent(),
            threshold_state = Silent(),
            forcing = Silent(),
            sensitivity_vjp_choice = Silent(),
        ),
        Standard = (
            linear_verbosity = None(),
            non_enclosing_interval = WarnLevel(),
            alias_u0_immutable = WarnLevel(),
            linsolve_failed_noncurrent = WarnLevel(),
            termination_condition = WarnLevel(),
            threshold_state = WarnLevel(),
            forcing = InfoLevel(),
            sensitivity_vjp_choice = WarnLevel(),
        ),
        Detailed = (
            linear_verbosity = Detailed(),
            non_enclosing_interval = WarnLevel(),
            alias_u0_immutable = WarnLevel(),
            linsolve_failed_noncurrent = WarnLevel(),
            termination_condition = WarnLevel(),
            threshold_state = WarnLevel(),
            forcing = InfoLevel(),
            sensitivity_vjp_choice = WarnLevel(),
        ),
        All = (
            linear_verbosity = Detailed(),
            non_enclosing_interval = WarnLevel(),
            alias_u0_immutable = WarnLevel(),
            linsolve_failed_noncurrent = WarnLevel(),
            termination_condition = WarnLevel(),
            threshold_state = InfoLevel(),
            forcing = InfoLevel(),
            sensitivity_vjp_choice = WarnLevel(),
        ),
    )

    groups = (
        error_control = (
            :non_enclosing_interval, :alias_u0_immutable,
            :linsolve_failed_noncurrent, :termination_condition,
        ),
        numerical = (:threshold_state, :forcing),
        sensitivity = (:sensitivity_vjp_choice,),
    )
end
