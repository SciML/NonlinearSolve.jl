# Test problems: (name, f, f', f'', bracket). `f(u)` without parameters; the benchmark
# scripts wrap it as needed.
const STANDARD = [
    ("u^2 - 2", u -> u^2 - 2, u -> 2u, u -> 2 + 0u, (1.0, 20.0)),
    ("exp(u) - 10", u -> exp(u) - 10, u -> exp(u), u -> exp(u), (0.0, 5.0)),
    ("cos(u) - u", u -> cos(u) - u, u -> -sin(u) - 1, u -> -cos(u), (0.0, 1.0)),
    ("u^3 - 2u - 5", u -> u^3 - 2u - 5, u -> 3u^2 - 2, u -> 6u, (2.0, 3.0)),
    ("log(u) - 1", u -> log(u) - 1, u -> 1 / u, u -> -1 / u^2, (1.0, 10.0)),
    (
        "0.3 - (e^-u + e^-3u)/2", u -> 0.3 - (exp(-u) + exp(-3u)) / 2,
        u -> (exp(-u) + 3exp(-3u)) / 2, u -> -(exp(-u) + 9exp(-3u)) / 2, (0.0, 5.0),
    ),
    ("u^10 - 1", u -> u^10 - 1, u -> 10u^9, u -> 90u^8, (0.0, 1.3)),
    ("atan(u) - 1", u -> atan(u) - 1, u -> 1 / (1 + u^2), u -> -2u / (1 + u^2)^2, (-10.0, 20.0)),
    ("(u-1)^3 - 1e-3", u -> (u - 1)^3 - 1.0e-3, u -> 3(u - 1)^2, u -> 6(u - 1), (0.0, 3.0)),
]

const HARD = [
    ("triple root (u-1)^3", u -> (u - 1)^3, u -> 3(u - 1)^2, u -> 6(u - 1), (0.0, 3.0)),
    ("5-fold root (u-1)^5", u -> (u - 1)^5, u -> 5(u - 1)^4, u -> 20(u - 1)^3, (0.0, 3.0)),
    ("flat u^20 - 1", u -> u^20 - 1, u -> 20u^19, u -> 380u^18, (0.0, 5.0)),
    ("Newton leaves: atan(u)", u -> atan(u), u -> 1 / (1 + u^2), u -> -2u / (1 + u^2)^2, (-10.0, 20.0)),
    (
        "Newton leaves: tanh(10(u-0.3))", u -> tanh(10(u - 0.3)), u -> 10sech(10(u - 0.3))^2,
        u -> -200sech(10(u - 0.3))^2 * tanh(10(u - 0.3)), (-1.0, 2.0),
    ),
    ("f'=0 at end: u^3 - 2", u -> u^3 - 2, u -> 3u^2, u -> 6u, (0.0, 2.0)),
    ("f'=0 at end: cos(u)", u -> cos(u), u -> -sin(u), u -> -cos(u), (0.0, 2.0)),
    ("kink: max(u, 3u-1) - 0.8", u -> max(u, 3u - 1) - 0.8, u -> u < 0.5 ? 1.0 : 3.0, u -> 0.0, (0.0, 1.0)),
    ("jump: sign(u - 1/3)", u -> u < 1 / 3 ? -1.0 : 1.0, u -> 0.0, u -> 0.0, (0.0, 1.0)),
]
