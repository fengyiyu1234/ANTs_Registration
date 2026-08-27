"""Read an ANTs run.log and report, per pyramid level, whether the iteration
budget was actually used -- i.e. whether raising `registration.reg_iterations`
would buy anything, and where the wall time is going.

Why a script: the obvious signal in the log is ANTs' own `convergenceValue`
against its 1e-7 threshold, and on this data that signal is misleading.
Measured on s12t / DeMBA_0825, the coarsest SyN level ended 100 iterations
with convergenceValue = 1.06e-5 -- two orders of magnitude "short" -- while its
last 20 iterations had gained only 4% of what its first 20 gained. CC on real
lightsheet data has a jitter floor that convergenceValue never drops below, so
a level can look starved and be entirely saturated. The metric TRAJECTORY is
what settles it, and that means parsing all several-thousand DIAGNOSTIC lines
rather than reading the tail.

What the columns mean:

* `smooth` is flow_sigma x shrink x voxel -- the physical width of the Gaussian
  applied to the update field at that level, i.e. the finest spatial detail the
  deformation can express there. Compare it against the size of the error you
  are trying to fix: a 552 um medial gap cannot be closed at a level whose
  smoothing is 60 um, no matter how many iterations it gets.
* `head`/`tail` are the metric gained over the first and last fifth of the
  level's iterations, off a running best (the metric is not monotone).
* `tail/head` is the descent-decay read, off the running best so it is never
  negative. It reads well while the descent is front-loaded, which every SyN
  level here is.
* `tail/avg` is what the verdict actually uses: the tail's gain-per-iteration
  over the level's own average gain-per-iteration. Below ~15% the level is done
  and extra iterations are wasted; above ~50% it was still descending when the
  budget ran out and is worth raising. Unlike `tail/head` it assumes nothing
  about the shape of the curve -- the Affine stage's meansquares metric does
  most of its work in the middle, where `tail/head` reports ~1% for a level
  that is genuinely converged.
* A level that drifted back UP from its own best is called out as OVERSHOT,
  read off the raw metric instead of the running best.
* `98%@` is the smallest iteration count that still captures `--target` of that
  level's total gain -- the number the suggestion at the bottom is built from.
  It is bounded by the budget that was actually run, so it can only ever say
  "you could have spent less", never "spend more": a level cut off mid-descent
  reports 98%@ = n-1 and looks done. `tail/avg` is the column that catches
  that case (measured: the shrink-1 level of DeMBA_0827, run at 25 iterations,
  reports 98%@ = 24 while tail/avg = 40% -- the highest of any level in any
  run here, i.e. the one level that genuinely wanted more).
* `conv` is ANTs' own convergenceValue at exit, printed only so it can be
  compared against `tail/head` and distrusted.

Usage (any env with numpy; no ANTs needed):

    python scripts/syn_convergence.py /path/to/output_dir/run.log
    python scripts/syn_convergence.py /path/to/output_dir          # finds run.log
    python scripts/syn_convergence.py old/run.log new/run.log      # compare runs
    python scripts/syn_convergence.py run.log --target 0.95 --all-stages

By default only the deformable (SyN) stage is reported, since that is the one
whose budget is worth tuning; --all-stages adds the Translation pre-align and
the shape-driven Affine.

IMPORTANT: the saturation numbers are a property of THIS metric landscape.
Adding or changing `mask.guide_regions` entries adds MeanSquares data terms and
changes it -- re-run this after any mask change instead of reusing an earlier
verdict.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

STAGE_RE = re.compile(r"\*\*\* Running (\w+) registration(?: \((.*?)\))? \*\*\*")
# The leading digits are the STAGE index within the antsRegistration call, not
# the level -- levels are delimited by the XXDIAGNOSTIC header line instead.
DIAG_RE = re.compile(
    r"^\s*\d+DIAGNOSTIC,\s*(\d+),\s*(-?[\d.]+e[-+]\d+),\s*(inf|-?[\d.]+e[-+]\d+),"
    r"\s*([\d.]+e[-+]\d+),\s*([\d.]+e[-+]\d+)")
HEADER_MARK = "DIAGNOSTIC,Iteration,metricValue"
SHRINK_RE = re.compile(r"Shrink factors \(level (\d+) out of (\d+)\):\s*\[(\d+)")
SIGMA_RE = re.compile(r"smoothing sigmas per level:\s*\[([^\]]*)\]")
FLOW_RE = re.compile(r"varianceForUpdateField\s*=\s*([\d.eE+-]+)")
VOXEL_RE = re.compile(r"Resampling fine level to ([\d.]+)\s*um isotropic")


def parse(path):
    """-> (voxel_um or None, [stage dicts]). One stage per antsRegistration
    call; each carries the shrink/sigma schedule printed just above its
    marker and one entry per pyramid level actually run."""
    voxel_um = None
    stages, stage = [], None
    pending_shrinks, pending_sigmas = {}, None
    for line in Path(path).read_text(errors="replace").splitlines():
        if voxel_um is None:
            m = VOXEL_RE.search(line)
            if m:
                voxel_um = float(m.group(1))
        m = SHRINK_RE.search(line)
        if m:
            pending_shrinks[int(m.group(1))] = int(m.group(3))
            continue
        m = SIGMA_RE.search(line)
        if m:
            pending_sigmas = [int(v) for v in m.group(1).replace(",", " ").split()]
            continue
        m = STAGE_RE.search(line)
        if m:
            flow = FLOW_RE.search(m.group(2) or "")
            stage = {
                "name": m.group(1),
                "flow_sigma": float(flow.group(1)) if flow else None,
                # printed 1-based, and the schedule is for the whole call --
                # a level that convergence exits before running never appears
                # in the diagnostics and is dropped when they are zipped below.
                "shrinks": [pending_shrinks[k] for k in sorted(pending_shrinks)],
                "sigmas": pending_sigmas or [],
                "levels": [],
            }
            stages.append(stage)
            pending_shrinks, pending_sigmas = {}, None
            continue
        if stage is None:
            continue
        if HEADER_MARK in line:
            stage["levels"].append({"it": [], "metric": [], "conv": [], "cum": [], "dt": []})
            continue
        m = DIAG_RE.match(line)
        if m and stage["levels"]:
            lv = stage["levels"][-1]
            lv["it"].append(int(m.group(1)))
            lv["metric"].append(float(m.group(2)))
            lv["conv"].append(float("nan") if m.group(3) == "inf" else float(m.group(3)))
            lv["cum"].append(float(m.group(4)))
            lv["dt"].append(float(m.group(5)))
    for st in stages:
        st["levels"] = [lv for lv in st["levels"] if lv["it"]]
    return voxel_um, [st for st in stages if st["levels"]]


def analyse(level, target):
    """Saturation read for one pyramid level.

    Everything is computed off a running minimum, not the raw metric: ANTs
    reports the metric at each iteration and it can rise again (the shrink-2
    SyN level here ends 3.4e-5 WORSE than 14 iterations earlier), which would
    otherwise make the tail gain look like progress with the wrong sign.
    """
    raw = np.asarray(level["metric"], float)
    m = np.minimum.accumulate(raw)
    n = len(m)
    gain = m[-1] - m[0]          # <= 0; more negative is better
    q = max(1, n // 5)
    head = m[q - 1] - m[0]
    tail = m[-1] - m[-q - 1] if n > q else gain
    if gain < 0:
        # smallest iteration count still worth paying for
        k = int(np.argmax((m - m[0]) <= target * gain)) + 1
    else:
        k = n
    # `tail/head` reads well when the descent is front-loaded (every SyN level
    # here is), but the Affine stage's meansquares metric does most of its work
    # in the MIDDLE, where that ratio is ~1% for a level that plainly converged.
    # The verdict is therefore built on tail rate vs the level's own average
    # rate, which assumes nothing about the shape of the curve: "at the speed it
    # was still going when the budget ran out, was it going anywhere?"
    avg_rate = gain / n
    tail_rate = tail / min(q, n)
    return {
        "n": n, "first": m[0], "last": m[-1], "gain": gain, "head": head, "tail": tail,
        # + 0.0 so an exactly-flat tail prints 0.0% rather than -0.0%
        "ratio": (tail / head + 0.0) if head < 0 else float("nan"),
        "rate": (tail_rate / avg_rate + 0.0) if avg_rate < 0 else float("nan"),
        # The running minimum above can never rise, so overshoot has to be read
        # off the RAW metric: how far the level drifted back up from its own
        # best before the budget ran out. Small here (1e-5 scale), but it is the
        # difference between "ran out of budget" and "ran past the answer".
        "overshoot": float(raw[-1] - m[-1]),
        "k": k,
        "conv": level["conv"][-1],
        "s_per_it": float(np.median(level["dt"])),
        "cum": level["cum"][-1],
    }


def report(path, target, all_stages):
    voxel_um, stages = parse(path)
    print(f"\n=== {path} ===")
    if not stages:
        print("  no ANTs DIAGNOSTIC output found -- this log was not produced by "
              "run_pipeline.sh (which tees ANTs' stdout), or verbose was off.")
        return
    print(f"  fine level: {voxel_um:g} um isotropic" if voxel_um else
          "  fine voxel size not found in log (smooth column omitted)")

    for st in stages:
        if not all_stages and "syn" not in st["name"].lower():
            continue
        flow = st["flow_sigma"]
        head_flow = f", flow sigma {flow:g} vox" if flow else ""
        print(f"\n  {st['name']}{head_flow} -- {len(st['levels'])} level(s) run")
        cols = (f"    {'lvl':>3} {'shrink':>6} {'voxel':>7} {'smooth':>8} {'iters':>5} "
                f"{'metric first -> last':>24} {'head/5':>10} {'tail/5':>10} "
                f"{'tail/head':>9} {'tail/avg':>8} {'98%@':>5} {'wall':>7} {'s/it':>7} "
                f"{'conv':>9}")
        print(cols)
        print("    " + "-" * (len(cols) - 4))
        rows, prev_cum = [], 0.0
        for i, lv in enumerate(st["levels"]):
            a = analyse(lv, target)
            shrink = st["shrinks"][i] if i < len(st["shrinks"]) else None
            vox = shrink * voxel_um if (shrink and voxel_um) else None
            smooth = flow * vox if (flow and vox) else None
            wall = a["cum"] - prev_cum
            prev_cum = a["cum"]
            a.update(shrink=shrink, vox=vox, smooth=smooth, wall=wall, i=i)
            rows.append(a)
            print(f"    {i:>3} {str(shrink or '?'):>6} "
                  f"{(f'{vox:g}um' if vox else '?'):>7} {(f'{smooth:g}um' if smooth else '?'):>8} "
                  f"{a['n']:>5} {a['first']:>11.6f} -> {a['last']:>9.6f} "
                  f"{a['head']:>10.5f} {a['tail']:>+10.5f} {pct(a['ratio']):>9} "
                  f"{pct(a['rate']):>8} {a['k']:>5} {fmt_t(wall):>7} {a['s_per_it']:>7.2f} "
                  f"{a['conv']:>9.1e}")
        interpret(rows, target, "syn" in st["name"].lower())


def pct(x):
    return "-" if x != x else f"{x:.0%}" if abs(x) >= 0.1 else f"{x:.1%}"


def fmt_t(sec):
    return f"{sec:.0f}s" if sec < 90 else (f"{sec / 60:.1f}m" if sec < 5400 else f"{sec / 3600:.2f}h")


def interpret(rows, target, is_syn):
    total = sum(r["wall"] for r in rows)
    print()
    for r in rows:
        share = r["wall"] / total if total else 0
        if r["gain"] >= 0:
            verdict = "NO PROGRESS -- this level never improved the metric"
        elif np.isnan(r["ratio"]):
            verdict = "too few iterations to read"
        elif r["overshoot"] > 0.1 * abs(r["gain"]):
            verdict = (f"OVERSHOT -- ended {r['overshoot']:.2e} worse than its own best; "
                       f"{r['k']} iters was enough")
        elif r["rate"] != r["rate"]:
            verdict = "too few iterations to read"
        elif r["rate"] < 0.15:
            verdict = f"saturated -- {r['k']} iters already captures {target:.0%} of its gain"
        elif r["rate"] < 0.50:
            verdict = "mostly done -- small return on more iterations"
        else:
            verdict = ("STILL DESCENDING at the budget limit -- raising this level is the one "
                       "place more iterations can help")
        print(f"    level {r['i']} ({share:5.1%} of stage wall time, {fmt_t(r['wall'])}): {verdict}")

    sug = [r["k"] for r in rows]
    proj = sum(k * r["s_per_it"] for k, r in zip(sug, rows))
    knob = "reg_iterations" if is_syn else "aff_iterations (NOT reg_iterations)"
    print(f"\n    {knob} for {target:.0%} of the same gain: "
          f"[{', '.join(str(k) for k in sug)}]"
          f"   ~{fmt_t(proj)} vs {fmt_t(total)} (excludes per-level setup)")
    starved = [r for r in rows if r["gain"] < 0 and r["rate"] > 0.50]
    if starved:
        print("    ...but raise level(s) " + ", ".join(str(r["i"]) for r in starved) +
              " above their current count first -- they had not converged.")
    if is_syn and rows and rows[0]["shrink"]:
        n = len(rows)
        print(f"\n    Adding a COARSER level is a different lever from more iterations: ANTsPy "
              f"derives\n    the pyramid from len(reg_iterations), so {n + 1} entries give "
              f"shrink {'x'.join(str(2 ** s) for s in range(n, -1, -1))} "
              f"instead of {'x'.join(str(2 ** s) for s in range(n - 1, -1, -1))}.")
        if rows[0]["smooth"]:
            print(f"    That prepends a level smoothing at {2 * rows[0]['smooth']:g} um "
                  f"(vs {rows[0]['smooth']:g} um now) for roughly "
                  f"{rows[0]['s_per_it'] / 8:.2f} s/iter.\n    Saturation here says nothing "
                  f"about whether that scale helps -- it is a different search\n    space, "
                  f"so it has to be tried.")
    if not is_syn:
        return
    print("\n    Saturation is a property of this metric landscape. Change mask.guide_regions "
          "and\n    re-measure -- each added MeanSquares term changes what the levels have to do.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", help="run.log path(s), or output dir(s) containing one")
    ap.add_argument("--target", type=float, default=0.98,
                    help="fraction of each level's gain the suggestion must keep (default 0.98)")
    ap.add_argument("--all-stages", action="store_true",
                    help="also report the Translation pre-align and the Affine stage")
    args = ap.parse_args()
    if not 0 < args.target <= 1:
        sys.exit("--target must be in (0, 1]")

    for raw in args.logs:
        p = Path(raw)
        if p.is_dir():
            p = p / "run.log"
        if not p.is_file():
            sys.exit(f"no such log: {p}")
        report(p, args.target, args.all_stages)
    print()


if __name__ == "__main__":
    main()
