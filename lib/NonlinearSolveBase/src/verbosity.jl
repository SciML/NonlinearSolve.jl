mutable struct NonlinearErrorControlVerbosity
    immutable_u0::Verbosity.Type
    non_enclosing_interval::Verbosity.Type
    non_forward_mode::Verbosity.Type
    ad_backend_incompatible::Verbosity.Type

    function NonlinearErrorControlVerbosity()
        new()
    end
end

function NonlinearErrorControlVerbosity(verbose::Verbosity.Type)
    @match verbose begin
        Verbosity.None() => NonlinearErrorControlVerbosity(fill(
            Verbosity.None(), length(fieldnames(NonlinearErrorControlVerbosity)))...)

        Verbosity.Info() => NonlinearErrorControlVerbosity(fill(
            Verbosity.Info(), length(fieldnames(NonlinearErrorControlVerbosity)))...)

        Verbosity.Warn() => NonlinearErrorControlVerbosity(fill(
            Verbosity.Warn(), length(fieldnames(NonlinearErrorControlVerbosity)))...)

        Verbosity.Error() => NonlinearNumericalVerbosity(fill(
            Verbosity.Error(), length(fieldnames(NonlinearErrorControlVerbosity)))...)

        Verbosity.Default() => NonlinearErrorControlVerbosity()

        Verbosity.Edge() => NonlinearErrorControlVerbosity()

        _ => @error "Not a valid choice for verbosity."
    end
end

mutable struct NonlinearPerformanceVerbosity
    function NonlinearPerformanceVerbosity()
        new()
    end
end

function NonlinearPerformanceVerbosity(verbose::Verbosity.Type)
    @match verbose begin
        Verbosity.None() => NonlinearPerformanceVerbosity(fill(
            Verbosity.None(), length(fieldnames(NonlinearPerformanceVerbosity)))...)

        Verbosity.Info() => NonlinearPerformanceVerbosity(fill(
            Verbosity.Info(), length(fieldnames(NonlinearPerformanceVerbosity)))...)

        Verbosity.Warn() => NonlinPerformanceVerbosity(fill(
            Verbosity.Warn(), length(fieldnames(NonlinearPerformanceVerbosity)))...)

        Verbosity.Error() => NonlinearPerformanceVerbosity(fill(
            Verbosity.Error(), length(fieldnames(NonlinearPerformanceVerbosity)))...)

        Verbosity.Default() => NonlinearPerformanceVerbosity()

        Verbosity.Edge() => NonlinearPerformanceVerbosity()

        _ => @error "Not a valid choice for verbosity."
    end
end

mutable struct NonlinearNumericalVerbosity
    function NonlinearNumericalVerbosity()
        new()
    end
end

function NonlinearNumericalVerbosity(verbose::Verbosity.Type)
    @match verbose begin
        Verbosity.None() => NonlinearNumericalVerbosity(fill(
            Verbosity.None(), length(fieldnames(NonlinearPerformanceVerbosity)))...)

        Verbosity.Info() => NonlinearNumericalVerbosity(fill(
            Verbosity.Info(), length(fieldnames(NonlinearPerformanceVerbosity)))...)

        Verbosity.Warn() => NonlinearNumericalVerbosity(fill(
            Verbosity.Warn(), length(fieldnames(NonlinearPerformanceVerbosity)))...)

        Verbosity.Error() => NonlinearNumericalVerbosity(fill(
            Verbosity.Error(), length(fieldnames(NonlinearPerformanceVerbosity)))...)

        Verbosity.Default() => NonlinearNumericalVerbosity()

        Verbosity.Edge() => NonlinearNumericalVerbosity()

        _ => @error "Not a valid choice for verbosity."
    end
end

struct NonlinearVerbosity{T} <: AbstractVerbositySpecifier{T}
    linear_verbosity

    error_control::NonlinearErrorControlVerbosity
    performance::NonlinearPerformanceVerbosity
    numerical::NonlinearNumericalVerbosity
end

function NonlinearVerbosity(verbose::Verbosity.Type)
    @match verbose begin
        Verbosity.Default() => NonlinearVerbosity{true}(
            Verbosity.Default(),
            NonlinearErrorControlVerbosity(Verbosity.Default()),
            NonlinearPerformanceVerbosity(Verbosity.Default()),
            NonlinearNumericalVerbosity(Verbosity.Default())
        )

        Verbosity.None() => NonlinearVerbosity{false}(
            Verbosity.None(),
            NonlinearErrorControlVerbosity(Verbosity.None()),
            NonlinearPerformanceVerbosity(Verbosity.None()),
            NonlinearNumericalVerbosity(Verbosity.None()))

        Verbosity.All() => NonlinearVerbosity{true}(
            Verbosity.All(),
            NonlinearErrorControlVerbosity(Verbosity.Info()),
            NonlinearPerformanceVerbosity(Verbosity.Info()),
            NonlinearNumericalVerbosity(Verbosity.Info())
        )

        _ => @error "Not a valid choice for verbosity."
    end
end

function NonlinearVerbosity(;
    error_control=Verbosity.Default(), performance=Verbosity.Default(),
    numerical=Verbosity.Default(), linear_verbosity = Verbosity.Default(), kwargs...)


    if error_control isa Verbosity.Type
        error_control_verbosity = NonlinearErrorControlVerbosity(error_control)
    else
        error_control_verbosity = error_control
    end

    if performance isa Verbosity.Type
        performance_verbosity = NonlinearPerformanceVerbosity(performance)
    else
        performance_verbosity = performance
    end

    if numerical isa Verbosity.Type
        numerical_verbosity = NonlinearNumericalVerbosity(numerical)
    else
        numerical_verbosity = numerical
    end

    if !isempty(kwargs)
        for (key, value) in pairs(kwargs)
            if hasfield(NonlinearErrorControlVerbosity, key)
                setproperty!(error_control_verbosity, key, value)
            elseif hasfield(NonlinearPerformanceVerbosity, key)
                setproperty!(performance_verbosity, key, value)
            elseif hasfield(NonlinearNumericalVerbosity, key)
                setproperty!(numerical_verbosity, key, value)
            else
                error("$key is not a recognized verbosity toggle.")
            end
        end
    end

    NonlinearVerbosity{true}(linear_verbosity, error_control_verbosity,
        performance_verbosity, numerical_verbosity)
end