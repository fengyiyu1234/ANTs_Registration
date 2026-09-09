#!/usr/bin/env bash
# Every figure for one stats run, in one go.
#
#   conda activate antsreg
#   ./scripts/make_figures.sh stats/configs/tsc_marker_ungated.yaml
#   CLASS=GFP_any ./scripts/make_figures.sh stats/configs/tsc_marker_ungated.yaml
#
# Output lands in <output.dir>/figures/ as PNG (slides) + PDF (vector, for
# Illustrator). Re-running overwrites; nothing is appended.
#
# By default EVERY cell class the config declared gets drawn. The class list and
# its order come off the config (group_stats.class_vocabulary), so the same
# command works on a marker run and on a MADM run without editing anything.
# Set CLASS to restrict to one -- useful when iterating, but the full sweep is
# the honest default: choosing which class to draw after seeing which one came
# out is the same p-hacking as choosing include_ids that way.
#
# Start from summary_Density.png. Fifty maps are unusable without an index, and
# that figure is the index: it says which class-by-level tiles have anything on
# them before you open any of them.
set -euo pipefail
CFG="${1:?usage: make_figures.sh <config.yaml> [region]}"
REGION="${2:-Cerebral cortex}"
CLASS="${CLASS:-}"
if [ -n "$CLASS" ]; then
  SCOPE=(--class-name "$CLASS")
else
  SCOPE=(--all-classes)
fi

echo "=== index"
for M in Density Count; do
  python -m stats.plot_effects --config "$CFG" --kind summary --metric "$M"
done

echo "=== bars (one panel per class, all classes already)"
python -m stats.plot_bars --config "$CFG" --preset all --region "$REGION"

echo "=== atlas maps: per sample + significance, every class, L3 and L5"
python -m stats.plot_atlas_panels --config "$CFG" --mode both \
  "${SCOPE[@]}" --metric Density --levels 3,5 --n-coronal 1

echo "=== atlas survey: 4 sections, every class, L5"
python -m stats.plot_atlas_panels --config "$CFG" --mode per-sample \
  "${SCOPE[@]}" --metric Density --level 5 --n-coronal 4

echo "=== volcano, every class"
python -m stats.plot_effects --config "$CFG" --kind volcano \
  "${SCOPE[@]}" --metric Density --levels 2,3,5

# The forest plot is not in this list because its x axis IS Hedges' g. Run it
# by hand when an effect-size figure is wanted:
#   python -m stats.plot_effects --config "$CFG" --kind forest \
#     --all-classes --metric Density --level 5 --top 20

echo "=== done"
