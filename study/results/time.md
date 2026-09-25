### A. Standard functions as they are, abstol = 0: median time per solve (ns)
| problem | ModAB | ITP | Brent | PR NewtonBisection | NewtonBisection | ModABNewton | NewtonBisection AD | ModABNewton AD |
|---|---|---|---|---|---|---|---|---|
| u^2 - 2 | 136 | 240 | 378 | 102 | 113 | 146 | 114 | 146 |
| exp(u) - 10 | 195 | 412 | 424 | 241 | 228 | 199 | 221 | 184 |
| cos(u) - u | 87 | 279 | 188 | 100 | 103 | 110 | 108 | 119 |
| u^3 - 2u - 5 | 104 | 221 | 151 | 82 | 81 | 93 | 78 | 76 |
| log(u) - 1 | 145 | 253 | 236 | 176 | 232 | 200 | 221 | 207 |
| 0.3 - (e^-u + e^-3u)/2 | 257 | 526 | 371 | 445 | 366 | 326 | 322 | 253 |
| u^10 - 1 | 158 | 442 | 254 | 198 | 227 | 277 | 245 | 287 |
| atan(u) - 1 | 333 | 560 | 476 | 275 | 300 | 263 | 301 | 293 |
| (u-1)^3 - 1e-3 | 141 | 438 | 313 | 243 | 232 | 158 | 232 | 153 |
| **total** | 1557 | 3371 | 2792 | 1861 | 1882 | 1772 | 1842 | 1718 |
### B. Expensive f (1000 multiply-adds, 0.86 μs per f), f' costs r times f, abstol = 0: total μs over the standard functions
| r | ModAB | ITP | Brent | PR NewtonBisection | NewtonBisection | ModABNewton |
|---|---|---|---|---|---|---|
| 0.0 | 116.9 | 148.4 | 267.8 | 100.4 | 99.9 | 99.8 |
| 0.1 | 115.1 | 145.1 | 267.7 | 102.0 | 100.7 | 101.6 |
| 0.25 | 115.1 | 145.1 | 267.6 | 111.9 | 108.0 | 107.3 |
| 0.5 | 115.7 | 145.1 | 267.7 | 136.2 | 125.8 | 121.3 |
| 1.0 | 115.2 | 145.1 | 267.6 | 184.7 | 161.3 | 149.3 |
### C. Cost of one ForwardDiff evaluation relative to one f evaluation
Mean over 1000 points of the bracket, summed in a loop (so the plain f may use SIMD).
f' uses a Dual; f' and f'' (for Halley) use a Dual of Duals.
| problem | f (ns) | f, f' (ns) | ratio | f, f', f'' (ns) | ratio |
|---|---|---|---|---|---|
| u^2 - 2 | 0.10 | 0.24 | 2.35 | 0.22 | 2.22 |
| exp(u) - 10 | 4.01 | 4.00 | 1.00 | 4.01 | 1.00 |
| cos(u) - u | 4.07 | 5.14 | 1.26 | 3.99 | 0.98 |
| u^3 - 2u - 5 | 0.16 | 0.26 | 1.65 | 9.43 | 59.74 |
| log(u) - 1 | 3.81 | 4.57 | 1.20 | 4.58 | 1.20 |
| 0.3 - (e^-u + e^-3u)/2 | 6.58 | 5.62 | 0.85 | 11.68 | 1.78 |
| u^10 - 1 | 3.43 | 4.28 | 1.25 | 15.98 | 4.66 |
| atan(u) - 1 | 5.87 | 6.03 | 1.03 | 9.08 | 1.55 |
| (u-1)^3 - 1e-3 | 0.11 | 0.24 | 2.15 | 0.24 | 2.15 |
