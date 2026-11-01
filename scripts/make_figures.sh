#!/bin/bash
# Render every manuscript figure and table from results/ and data/ into
# figures/output/. Takes about two minutes on a laptop, most of it the
# model-comparison fit for Supplementary Fig. S2 / Table S2.
#
#   bash scripts/make_figures.sh            # all
#   bash scripts/make_figures.sh fig3_retrocue fig4_bias
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
ALL=(fig1_concept fig2_setsize fig3_retrocue fig4_bias fig5_circuit
     figS_gp_prior figS_model_comparison tables stats_report)
TARGETS=("${@:-${ALL[@]}}")

for name in "${TARGETS[@]}"; do
    echo "== $name"
    "$PYTHON" -m "figures.$name"
done
echo "Outputs in figures/output/"
