"""Render a family-weighted performance profile from the bounded benchmark CSVs."""
import math
import sys
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from analyze_bounded import NATIVE, load_rows

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['svg.hashsalt'] = 'bounded-solvers'
rows = load_rows(sys.argv[2:])
labels = {
    'bounded_default': 'Bounded default',
    'bounded_trust_region': 'Bounded trust region',
    'gn_linesearch': 'Projected Gauss–Newton',
    'reflective': 'Reflective trust region',
    'legacy_default': 'Previous default (transformed)',
}
cases = sorted({case for case, _ in rows if all((case, name) in rows for name in labels)})
families = Counter(rows[case, 'bounded_default']['family'] for case in cases)
weights = {case: 1 / len(families) / families[rows[case, 'bounded_default']['family']] for case in cases}
fastest = {case: min((float(rows[case, name]['time_ns']) for name in NATIVE
                     if rows[case, name]['passed'] == 'true'), default=math.inf) for case in cases}
thresholds = np.geomspace(1, 1000, 500)
fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
for name, label in labels.items():
    fractions = [sum(weights[case] for case in cases if rows[case, name]['passed'] == 'true'
                     and float(rows[case, name]['time_ns']) <= tau * fastest[case]) for tau in thresholds]
    ax.plot(thresholds, fractions, label=label, linewidth=2.5 if name == 'bounded_default' else 1.6)
ax.set(xscale='log', xlim=(1, 1000), ylim=(0, 1),
       xlabel='Time / fastest verified native method',
       ylabel='Fraction of cases (equal family weights)',
       title=f'Bounded solves: {len(cases)} cases, {len(families)} families')
ax.grid(alpha=.2)
ax.legend(loc='lower right', framealpha=.95, fontsize=9)
output = Path(sys.argv[1])
output.parent.mkdir(parents=True, exist_ok=True)
if output.suffix == '.svg':
    fig.savefig(output, metadata={'Date': None})
    output.write_text('\n'.join(line.rstrip() for line in output.read_text().splitlines()) + '\n')
else:
    fig.savefig(output)
