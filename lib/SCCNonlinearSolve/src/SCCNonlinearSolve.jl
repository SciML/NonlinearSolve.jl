module SCCNonlinearSolve

import SciMLBase
import CommonSolve

function CommonSolve.solve(prob::SciMLBase.SCCNonlinearProblem, alg; kwargs...)
	numscc = length(prob.probs)
	sols = [SciMLBase.build_solution(prob, nothing, prob.u0, convert(eltype(prob.u0),NaN)*prob.u0) for prob in prob.probs]
	u = reduce(vcat,[prob.u0 for prob in prob.probs])
	resid = copy(u)

    lasti = 1
	for i in 1:numscc
		prob.explictfuns![i](parameter_values(prob.probs[i])[1],sols)
		sol = SciMLBase.solve(prob.probs[i], alg; kwargs...)
		_sol = SciMLBase.build_solution(prob.probs[i], nothing, sol.u, sol.resid, retcode = sol.retcode)
		sols[i] = _sol
                lasti = i
        if !SciMLBase.successful_retcode(_sol)
            break
        end
	end
	
	# TODO: fix allocations with a lazy concatenation
	u .= reduce(vcat,sols)
	resid .= reduce(vcat,getproperty.(sols,:resid))

        retcode = sols[lasti].retcode

	SciMLBase.build_solution(prob, alg, u, resid; retcode, original = sols)
end

end