function SciMLBase.isautodifferentiable(alg::SimpleIDSolve)
    true
end
function SciMLBase.allows_arbitrary_number_types(alg::SimpleIDSolve)
    true
end
function SciMLBase.allowscomplex(alg::SimpleIDSolve)
    true
end

SciMLBase.isdiscrete(alg::SimpleIDSolve) = true

isfsal(alg::SimpleIDSolve) = false
alg_order(alg::SimpleIDSolve) = 0
beta2_default(alg::SimpleIDSolve) = 0
beta1_default(alg::SimpleIDSolve, beta2) = 0

dt_required(alg::SimpleIDSolve) = false
isdiscretealg(alg::SimpleIDSolve) = true

function DiffEqBase._initialize_dae!(integrator, prob::ImplicitDiscreteProblem,
        alg::DefaultInit, x::Union{Val{true}, Val{false}})
    atol = one(eltype(prob.u0)) * 1e-12
    if SciMLBase.has_initializeprob(prob.f)
        _initialize_dae!(integrator, prob,
                         OverrideInit(atol), x)
    elseif !applicable(_initialize_dae!, integrator, prob,
        BrownFullBasicInit(atol), x)
        error("`OrdinaryDiffEqNonlinearSolve` is not loaded, which is required for the default initialization algorithm (`BrownFullBasicInit` or `ShampineCollocationInit`). To solve this problem, either do `using OrdinaryDiffEqNonlinearSolve` or pass `initializealg = CheckInit()` to the `solve` function. This second option requires consistent `u0`.")
    else
        _initialize_dae!(integrator, prob,
            BrownFullBasicInit(atol), x)
    end
end
