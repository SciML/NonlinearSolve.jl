"""Independently reproduce the nonzero least-squares reference costs using SciPy."""
import json
import numpy as np
from scipy.optimize import least_squares

bard_y = np.array([.14, .18, .22, .25, .29, .32, .35, .39, .37, .58, .73, .96, 1.34, 2.1, 4.39])
i = np.arange(1, 16)
bard = lambda u: bard_y - u[0] - i / (u[1] * (16 - i) + u[2] * np.minimum(i, 16 - i))
x = np.array([4., 2., 1., .5, .25, .167, .125, .1, .0833, .0714, .0625])
y = np.array([.1957, .1947, .1735, .16, .0844, .0627, .0456, .0342, .0323, .0235, .0246])
kowalik = lambda u: y - u[0] * (x**2 + u[1] * x) / (x**2 + u[2] * x + u[3])
results = {}
for name, fun, start, bounds in (
    ('bard', bard, [1., 1., 1.], ([0., .01, .01], [5., 10., 10.])),
    ('kowalik', kowalik, [.25, .39, .415, .39], ([.001] * 4, [5.] * 4)),
):
    sol = least_squares(fun, start, bounds=bounds, jac='3-point',
                        gtol=1e-14, ftol=1e-14, xtol=1e-14, max_nfev=10000)
    assert sol.success
    results[name] = dict(solution=sol.x.tolist(), cost=sol.cost, optimality=sol.optimality)
print(json.dumps(results, indent=2))
