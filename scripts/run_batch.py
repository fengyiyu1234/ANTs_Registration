"""Run the registration pipeline for configs/batch/*.yaml, a few at a time.

Skips configs marked NOT READY (soma image not stitched yet) and samples whose
output_dir already has results (same rule as run_pipeline.sh). Each run's
stdout/stderr -- including ANTs' own per-iteration output -- goes to
<output_dir>/run.log. Peak memory is ~5x the soma TIFF size (uint16 read +
float32 copy + transpose), i.e. up to ~75 GB for the largest samples, hence
the low default for --jobs.

    python scripts/run_batch.py [--only NAME ...] [--jobs 3] [--dry-run]
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch_samples as bs  # noqa: E402
from registration_ants.config import load_config  # noqa: E402


def pending(only):
    todo = []
    samples = ([s for n in only for s in bs.SAMPLES if s["name"] == n] if only else bs.SAMPLES)
    for s in samples:
        path = bs.CONFIG_OUT_DIR / f"{s['name']}.yaml"
        if not path.exists():
            print(f"{s['name']}: no config, run make_batch_configs.py first")
            continue
        if "NOT READY" in path.read_text(encoding="utf-8").split("\n", 1)[0]:
            print(f"{s['name']}: NOT READY, skipped")
            continue
        out = Path(load_config(path)["output_dir"])
        if out.is_dir() and any(p.name != "run.log" for p in out.iterdir()):
            print(f"{s['name']}: output exists ({out}), skipped")
            continue
        todo.append((s["name"], path, out))
    return todo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*")
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    todo = pending(args.only)
    print(f"{len(todo)} to run: {[t[0] for t in todo]}")
    if args.dry_run:
        return
    running, failed = [], []
    while todo or running:
        while todo and len(running) < args.jobs:
            name, cfg, out = todo.pop(0)
            out.mkdir(parents=True, exist_ok=True)
            log = open(out / "run.log", "a", encoding="utf-8")
            proc = subprocess.Popen([sys.executable, "-u", "-m", "registration_ants.pipeline", str(cfg)],
                                    stdout=log, stderr=subprocess.STDOUT)
            running.append((name, proc, log, time.time()))
            print(f"started {name} (pid {proc.pid}) -> {out / 'run.log'}", flush=True)
        time.sleep(20)
        for item in list(running):
            name, proc, log, t0 = item
            if proc.poll() is not None:
                log.close()
                running.remove(item)
                mins = (time.time() - t0) / 60
                print(f"finished {name}: exit {proc.returncode} after {mins:.0f} min", flush=True)
                if proc.returncode:
                    failed.append(name)
    print("all done" + (f"; FAILED: {failed}" if failed else ""))


if __name__ == "__main__":
    main()
