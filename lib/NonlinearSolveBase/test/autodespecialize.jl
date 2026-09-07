using ADTypes: AutoEnzyme, AutoForwardDiff
using NonlinearSolveBase, SciMLBase, Test

struct DynamicParameters
    rate::Float64
end

struct OtherDynamicParameters
    rate::Float64
    unused::Int
end

const seen_residual_parameter = Ref{DataType}()
const seen_jacobian_parameter = Ref{DataType}()
const seen_oop_parameter = Ref{DataType}()

function dynamic_residual!(resid, u, p)
    seen_residual_parameter[] = typeof(p)
    resid[1] = u[1]^2 - p.rate
    return nothing
end

function dynamic_jacobian!(J, u, p)
    seen_jacobian_parameter[] = typeof(p)
    J[1, 1] = 2u[1]
    return nothing
end

function dynamic_problem(p)
    f = NonlinearFunction{true, SciMLBase.AutoDespecialize}(
        dynamic_residual!; jac = dynamic_jacobian!
    )
    return NonlinearProblem(f, [1.0], p)
end

function dynamic_residual(u, p)
    seen_oop_parameter[] = typeof(p)
    return [u[1]^2 - p.rate]
end

function oop_dynamic_problem(p)
    f = NonlinearFunction{false, SciMLBase.AutoDespecialize}(dynamic_residual)
    return NonlinearProblem(f, [1.0], p)
end

@testset "AutoDespecialize dynamic parameter barrier" begin
    first_prob = NonlinearSolveBase.get_concrete_problem(
        dynamic_problem(DynamicParameters(2.0))
    )
    second_prob = NonlinearSolveBase.get_concrete_problem(
        dynamic_problem(OtherDynamicParameters(3.0, 1))
    )

    @test first_prob.p isa SciMLBase.DespecializedParameters
    @test SciMLBase.unwrap_parameters(first_prob.p) isa DynamicParameters
    @test SciMLBase.unwrap_parameters(second_prob.p) isa OtherDynamicParameters
    @test typeof(first_prob.p) === typeof(second_prob.p)
    @test typeof(first_prob.f) === typeof(second_prob.f)
    @test typeof(first_prob) === typeof(second_prob)

    resid = zeros(1)
    first_prob.f(resid, [2.0], first_prob.p)
    @test resid == [2.0]
    @test seen_residual_parameter[] === DynamicParameters

    J = zeros(1, 1)
    first_prob.f.jac(J, [2.0], first_prob.p)
    @test J == [4.0;;]
    @test seen_jacobian_parameter[] === DynamicParameters

    remade = NonlinearSolveBase.get_concrete_problem(first_prob)
    @test typeof(remade) === typeof(first_prob)
    @test remade.p === first_prob.p

    @test first_prob.f.f !== dynamic_residual!
    enzyme_prob = NonlinearSolveBase.maybe_unwrap_prob_for_enzyme(first_prob, AutoEnzyme())
    @test enzyme_prob.p isa DynamicParameters
    @test enzyme_prob.f.f === dynamic_residual!
end

@testset "Jacobian cache preserves raw parameter storage" begin
    p = DynamicParameters(2.0)
    prob = dynamic_problem(p)
    fu = zeros(1)
    prob.f(fu, prob.u0, p)
    cache = NonlinearSolveBase.construct_jacobian_cache(
        prob, nothing, prob.f, fu;
        stats = SciMLBase.NLStats(0, 0, 0, 0, 0),
        autodiff = AutoForwardDiff(),
    )

    wrapped = SciMLBase.DespecializedParameters(p)
    NonlinearSolveBase.InternalAPI.reinit!(cache; p = wrapped)
    @test cache.p === p
end

@testset "AutoDespecialize out-of-place residual" begin
    first_prob = NonlinearSolveBase.get_concrete_problem(
        oop_dynamic_problem(DynamicParameters(2.0))
    )
    second_prob = NonlinearSolveBase.get_concrete_problem(
        oop_dynamic_problem(OtherDynamicParameters(3.0, 1))
    )

    @test typeof(first_prob) === typeof(second_prob)
    @test first_prob.p isa SciMLBase.DespecializedParameters
    @test first_prob.f([2.0], first_prob.p) == [2.0]
    @test seen_oop_parameter[] === DynamicParameters
end

@testset "AutoDespecialize nonlinear problem variants" begin
    problem_builders = (
        p -> NonlinearLeastSquaresProblem(dynamic_problem(p).f, [1.0], p),
        p -> SciMLBase.ImmutableNonlinearProblem(dynamic_problem(p)),
    )
    for problem_builder in problem_builders
        first_prob = NonlinearSolveBase.get_concrete_problem(
            problem_builder(DynamicParameters(2.0))
        )
        second_prob = NonlinearSolveBase.get_concrete_problem(
            problem_builder(OtherDynamicParameters(3.0, 1))
        )
        @test typeof(first_prob) === typeof(second_prob)
        @test first_prob.p isa SciMLBase.DespecializedParameters
    end
end

@testset "other specialization policies retain concrete parameters" begin
    p = DynamicParameters(2.0)
    for specialize in (SciMLBase.AutoSpecialize, SciMLBase.FullSpecialize)
        f = NonlinearFunction{true, specialize}(dynamic_residual!)
        prob = NonlinearSolveBase.get_concrete_problem(NonlinearProblem(f, [1.0], p))
        @test prob.p === p
    end
end

@testset "wrapped callbacks report the wrapped function's arity" begin
    using Enzyme
    prob = NonlinearSolveBase.get_concrete_problem(dynamic_problem(DynamicParameters(2.0)))
    @test prob.f.f isa NonlinearSolveBase.AutoSpecializeCallable
    @test prob.f.jac isa NonlinearSolveBase.ParameterDespecializationWrapper
    @test SciMLBase.numargs(prob.f.f) == SciMLBase.numargs(dynamic_residual!)
    @test SciMLBase.numargs(prob.f.jac) == SciMLBase.numargs(dynamic_jacobian!)
    @test SciMLBase.isinplace(prob.f.jac, 3, "jac", true)

    # `remake` rebuilds the `NonlinearFunction`, which re-derives `isinplace` for every
    # callback; that must stay differentiable when the wrapped problem is remade under
    # Enzyme reverse mode.
    function remake_loss(u, prob)
        new_prob = remake(prob; u0 = prob.u0 .* u[1])
        return sum(new_prob.u0)
    end
    u = [1.5]
    du = zero(u)
    Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.Reverse), Enzyme.Const(remake_loss),
        Enzyme.Active, Enzyme.Duplicated(u, du), Enzyme.Const(prob)
    )
    @test du[1] ≈ sum(prob.u0)
end
