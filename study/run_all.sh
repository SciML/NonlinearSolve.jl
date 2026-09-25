#!/bin/bash
# Runs every benchmark of the study and writes the tables to study/results/.
#   bash study/run_all.sh
# Needs the environments study/ and study/ode/ instantiated (see REPORT.md), and for
# loadtime.jl the two environments named in REPORT.md.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results
J="julia +1.12"
strip_logs() { grep -v -E '^\s*[0-9.]+ ms|Precompil|dependenc|^\s*$' || true; }
$J explore.jl 2>&1 | sed "s/[[:space:]]*$//" > results/proto_counts.md
$J --project=. bench_counts.jl 2>&1 | strip_logs > results/counts.md
$J --project=. bench_time.jl 2>&1 | strip_logs > results/time.md
$J --project=ode ode/bench_ode.jl 2>&1 | strip_logs > results/ode.md
$J --project=. discriminate.jl > results/discriminate.txt 2>&1 || true
$J --project=. multiple_roots.jl 2>&1 | strip_logs > results/multiple_roots.md
$J cost_units.jl > results/cost_units.md
# then: julia study/loadtime.jl pr1330=<env with PR 1330> branch=<env with this branch>
