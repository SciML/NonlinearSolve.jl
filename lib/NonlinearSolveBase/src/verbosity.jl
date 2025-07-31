nonlinear_verbosity_defaults = Dict(
    :immutable_u0 => Verbosity.Warn(),
    :non_enclosing_interval => Verbosity.Warn(),
    :non_forward_mode => Verbosity.Warn(),
    :fd_ad_caution => Verbosity.Warn(),
    :ad_backend_incompatible => Verbosity.Warn(),
    :colorvec_non_sparse => Verbosity.Warn(),
    :colorvec_no_prototype => Verbosity.Warn(),
    :sparsity_using_jac_prototype => Verbosity.Warn(),
    :sparse_matrixcolorings_not_loaded => Verbosity.Warn(),
    :alias_u0_immutable => Verbosity.Warn(),
    :linsolve_failed_noncurrent => Verbosity.Warn(),
    :jacobian_free => Verbosity.Warn(),
    :termination_condition => Verbosity.Warn(),
    :threshold_state => Verbosity.Warn(),
    :pinv_undefined => Verbosity.Warn()
)


mutable struct NonlinearErrorControlVerbosity
    immutable_u0::Verbosity.Type
    non_enclosing_interval::Verbosity.Type
    non_forward_mode::Verbosity.Type
    fd_ad_caution::Verbosity.Type
    ad_backend_incompatible::Verbosity.Type
    alias_u0_immutable::Verbosity.Type
    linsolve_failed_noncurrent::Verbosity.Type
    jacobian_free::Verbosity.Type
    termination_condition::Verbosity.Type

    function NonlinearErrorControlVerbosity(immutable_u0 = nonlinear_verbosity_defaults[:immutable_u0],
        non_enclosing_interval = nonlinear_verbosity_defaults[:non_enclosing_interval],
        non_forward_mode = nonlinear_verbosity_defaults[:non_forward_mode],
        fd_ad_caution = nonlinear_verbosity_defaults[:fd_ad_caution],
        ad_backend_incompatible = nonlinear_verbosity_defaults[:ad_backend_incompatible],
        alias_u0_immutable = nonlinear_verbosity_defaults[:alias_u0_immutable],
        linsolve_failed_noncurrent = nonlinear_verbosity_defaults[:linsolve_failed_noncurrent],
        jacobian_free = nonlinear_verbosity_defaults[:jacobian_free],
        termination_condition = nonlinear_verbosity_defaults[:termination_condition])
        new(immutable_u0, non_enclosing_interval, non_forward_mode, fd_ad_caution, ad_backend_incompatible,
        alias_u0_immutable, linsolve_failed_noncurrent, jacobian_free, termination_condition)
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

        Verbosity.Error() => NonlinearErrorControlVerbosity(fill(
            Verbosity.Error(), length(fieldnames(NonlinearErrorControlVerbosity)))...)

        Verbosity.Default() => NonlinearErrorControlVerbosity()

        Verbosity.Edge() => NonlinearErrorControlVerbosity()

        _ => @error "Not a valid choice for verbosity."
    end
end

mutable struct NonlinearPerformanceVerbosity
    colorvec_non_sparse::Verbosity.Type
    colorvec_no_prototype::Verbosity.Type
    sparsity_using_jac_prototype::Verbosity.Type
    sparse_matrixcolorings_not_loaded::Verbosity.Type

    function NonlinearPerformanceVerbosity(colorvec_non_sparse = nonlinear_verbosity_defaults[:colorvec_non_sparse],
            colorvec_no_prototype = nonlinear_verbosity_defaults[:colorvec_no_prototype],
            sparsity_using_jac_prototype = nonlinear_verbosity_defaults[:sparsity_using_jac_prototype],
            sparse_matrixcolorings_not_loaded = nonlinear_verbosity_defaults[:sparse_matrixcolorings_not_loaded])
        new(colorvec_non_sparse, colorvec_no_prototype, sparsity_using_jac_prototype, sparse_matrixcolorings_not_loaded)
    end
end

function NonlinearPerformanceVerbosity(verbose::Verbosity.Type)
    @match verbose begin
        Verbosity.None() => NonlinearPerformanceVerbosity(fill(
            Verbosity.None(), length(fieldnames(NonlinearPerformanceVerbosity)))...)

        Verbosity.Info() => NonlinearPerformanceVerbosity(fill(
            Verbosity.Info(), length(fieldnames(NonlinearPerformanceVerbosity)))...)

        Verbosity.Warn() => NonlinearPerformanceVerbosity(fill(
            Verbosity.Warn(), length(fieldnames(NonlinearPerformanceVerbosity)))...)

        Verbosity.Error() => NonlinearPerformanceVerbosity(fill(
            Verbosity.Error(), length(fieldnames(NonlinearPerformanceVerbosity)))...)

        Verbosity.Default() => NonlinearPerformanceVerbosity()

        Verbosity.Edge() => NonlinearPerformanceVerbosity()

        _ => @error "Not a valid choice for verbosity."
    end
end

mutable struct NonlinearNumericalVerbosity
    threshold_state::Verbosity.Type
    pinv_undefined::Verbosity.Type
    function NonlinearNumericalVerbosity(;
        threshold_state = nonlinear_verbosity_defaults[:threshold_state], 
        pinv_undefined = nonlinear_verbosity_defaults[:pinv_undefined])
        new(threshold_state, pinv_undefined)
    end
end

function NonlinearNumericalVerbosity(verbose::Verbosity.Type)
    @match verbose begin
        Verbosity.None() => NonlinearNumericalVerbosity(fill(
            Verbosity.None(), length(fieldnames(NonlinearNumericalVerbosity)))...)

        Verbosity.Info() => NonlinearNumericalVerbosity(fill(
            Verbosity.Info(), length(fieldnames(NonlinearNumericalVerbosity)))...)

        Verbosity.Warn() => NonlinearNumericalVerbosity(fill(
            Verbosity.Warn(), length(fieldnames(NonlinearNumericalVerbosity)))...)

        Verbosity.Error() => NonlinearNumericalVerbosity(fill(
            Verbosity.Error(), length(fieldnames(NonlinearNumericalVerbosity)))...)

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