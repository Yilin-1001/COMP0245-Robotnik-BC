"""Network size: f-DAgger + Gate at 32x3, 64x3, 128x3 and 256x3 hidden layers. Rebuilds ../tables from ../data.

Sources:
  32x3 / 64x3 / 128x3 f-DAgger + Gate: teammate Hangyu Wu's runs (branch feature/small-net-dagger, his machine),
      one seed-0 DAgger chain per width, 221 fitting episodes per round (no development scenes, as the report's v3),
      round chosen on seed-0 validation, seeds 0-2. Only his aggregated results are available
      (data/teammate/summary.csv and <arch>/summary.csv): per-seed validation counts and pooled test counts per group.
  256x3 f-DAgger + Gate: the report's v3 R8 (per-scene validation and unified-test rollouts, this machine).
  OU + Gate per width: 32x3/64x3/128x3 from the 2026-11-02 architecture ablation (this machine, per-architecture lr);
      256x3 is the report's v1 R4. These are NOT the round-0 students of the teammate's chains (he retrained his own,
      2026-11-04, seed 0 only), so the OU + Gate -> f-DAgger + Gate gain per width mixes two training runs.

Tables:
  naming.csv      every model: report name, size label, canonical run id (Report naming), original run id, source
  validation.csv  32-scene validation sustained hold, seeds 0-2 pooled (n=96), 95% Wilson interval, per seed
  test.csv        240-scene unified test, seeds 0-2 pooled, grouped as in the teammate's summary
                  (in range T0+T4+T5, edges T2+T3, far T1, all 240), 95% Wilson interval
  efficiency.csv  parameters, FLOPs, .npz size, NumPy latency, training and DAgger-pipeline minutes
  rerun.csv       teammate models re-validated on this machine vs his recorded validation (if data/validation/rerun)
  network-size.tex  LaTeX table for the paper (Results, MLP Size)
  summary.md      all of the above as Markdown

Wilson intervals treat the pooled rollouts as independent; the three seeds share scenes, so they understate the
uncertainty. Scene-cluster bootstraps (as in 06/08) need per-scene records, which exist for 256x3 only.
No simulation or training runs here.

Usage (from final/robotnik_arm_task/):
    python report/09_network_size/code/make_tables.py
"""
import csv
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
DATA, OUT = HERE / "data", HERE / "tables"
INPUTS, OUTPUTS = 22, 9
SIZES = {  # architecture id -> (hidden widths, size label)
    "MLP02": ([32, 32, 32], "32x3"),
    "MLP04": ([64, 64, 64], "64x3"),
    "MLP06": ([128, 128, 128], "128x3"),
    "MLP01": ([256, 256, 256], "256x3"),
}
ORDER = ["MLP02", "MLP04", "MLP06", "MLP01"]
GROUPS = {"in range (T0+T4+T5)": {"T0", "T4", "T5"}, "edges (T2+T3)": {"T2", "T3"}, "far (T1)": {"T1"},
          "all 240": {"T0", "T1", "T2", "T3", "T4", "T5"}}
OURS_R8 = "R8-dart-gate-dagger-nodev-r3__MLP01-256x3__s{s}__v3__2026-11-05"
OURS_R4 = {"MLP01": "R4-dart-gate__MLP01-256x3__s{s}__v1__2026-10-28",
           "MLP02": "R4-dart-gate__MLP02-32x3__s{s}__v1__2026-11-02",
           "MLP04": "R4-dart-gate__MLP04-64x3__s{s}__v1__2026-11-02",
           "MLP06": "R4-dart-gate__MLP06-128x3__s{s}__v1__2026-11-02"}
X = "×"


def size_label(arch: str) -> str:
    return SIZES[arch][1].replace("x", X)


def arch_token(arch: str) -> str:
    return f"{arch}-{SIZES[arch][1]}"


def canonical(row: str, arch: str, seed: int, round_: int | None, version: str, date: str) -> str:
    """Report naming (docs/model-naming.md): R4-OU-Gate / R8-f-DAgger-Gate-r<k>, architecture, seed, version, date."""
    stem = "R4-OU-Gate" if row == "R4" else f"R8-f-DAgger-Gate-r{round_}"
    return f"{stem}__{arch_token(arch)}__s{seed}__{version}__{date}"


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "pct": round(100 * k / n, 1), "lo": round(lo, 1), "hi": round(hi, 1)}


def fmt(r: dict) -> str:
    return f"{r['pct']:.1f} [{r['lo']:.1f}, {r['hi']:.1f}] ({r['k']}/{r['n']})"


def read(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def mlp_cost(hidden: list[int]) -> tuple[int, int]:
    """Same count as evaluation/latency_bench.py: weights, biases and LayerNorm gamma/beta; FLOPs = 2 x MACs."""
    params, flops, width = 0, 0, INPUTS
    for h in hidden:
        params += width * h + h + 2 * h
        flops += 2 * width * h
        width = h
    return params + width * OUTPUTS + OUTPUTS, flops + 2 * width * OUTPUTS


def holds(run: str) -> tuple[int, int]:
    rows = read(DATA / "validation/per_scene" / f"{run}.csv")
    return sum(r["sustained_hold_pass"] == "True" for r in rows), len(rows)


def teammate() -> dict:
    """Per architecture: chosen round, per-seed validation holds, round-0 seed-0 holds, run ids, test counts, costs."""
    top = {r["arch"]: r for r in read(DATA / "teammate/summary.csv")}
    out = {}
    for arch in ["MLP02", "MLP04", "MLP06"]:
        rows = read(DATA / f"teammate/{arch}/summary.csv")
        k = int(top[arch]["chosen_round"])
        final = {int(r["seed"]): r for r in rows if int(r["round"]) == k}
        r0 = next(r for r in rows if int(r["round"]) == 0)
        test = {}
        for g in GROUPS:
            m = re.search(r"\((\d+)/(\d+)\)", top[arch][f"test {g}"])
            test[g] = (int(m.group(1)), int(m.group(2)))
        out[arch] = {
            "round": k,
            "seeds": {s: tuple(map(int, final[s]["sustained_hold"].split("/"))) for s in sorted(final)},
            "runs": {s: final[s]["run"] for s in sorted(final)},
            "r0_run": r0["run"], "r0_hold": tuple(map(int, r0["sustained_hold"].split("/"))),
            "by_round_s0": top[arch]["validation_by_round_s0"],
            "test": test, "top": top[arch],
        }
    expert = {}
    exp = next(r for r in read(DATA / "teammate/summary.csv") if r["arch"] == "teacher A")
    for g in GROUPS:
        m = re.search(r"\((\d+)/(\d+)\)", exp[f"test {g}"])
        expert[g] = (int(m.group(1)), int(m.group(2)))
    out["expert"] = expert
    return out


def local_test() -> dict:
    rows = read(DATA / "test/mlp01_rollouts.csv")
    ok = lambda r: r["success"] in ("True", "true", "1")
    res = {}
    for key, sel in {"R8": lambda r: r["row"] == "R8", "R4": lambda r: r["row"] == "R4",
                     "expert": lambda r: r["row"] == "expert"}.items():
        rs = [r for r in rows if sel(r)]
        res[key] = {g: (sum(ok(r) for r in rs if r["suite"] in s), sum(r["suite"] in s for r in rs))
                    for g, s in GROUPS.items()}
    return res


def write_csv(name: str, rows: list[dict]) -> None:
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    OUT.mkdir(exist_ok=True)
    tm, lt = teammate(), local_test()
    seeds = [0, 1, 2]

    # ---------------------------------------------------------------- naming
    naming = []
    for arch in ORDER:
        if arch == "MLP01":
            for s in seeds:
                naming.append({"report_name": "f-DAgger + Gate", "size": size_label(arch), "row": "R8", "round": 3,
                               "seed": s, "run_id": canonical("R8", arch, s, 3, "v3", "2026-11-05"),
                               "original_run_id": OURS_R8.format(s=s),
                               "source": "report v3 (this machine; checkpoints/2026-11-05final)"})
        else:
            t = tm[arch]
            for s in seeds:
                naming.append({"report_name": "f-DAgger + Gate", "size": size_label(arch), "row": "R8",
                               "round": t["round"], "seed": s,
                               "run_id": canonical("R8", arch, s, t["round"], "v3", "2026-11-05"),
                               "original_run_id": t["runs"][s],
                               "source": "teammate, feature/small-net-dagger (his machine)"})
    for arch in ORDER:
        date = "2026-10-28" if arch == "MLP01" else "2026-11-02"
        for s in seeds:
            naming.append({"report_name": "OU + Gate", "size": size_label(arch), "row": "R4", "round": 0, "seed": s,
                           "run_id": canonical("R4", arch, s, None, "v1", date),
                           "original_run_id": OURS_R4[arch].format(s=s),
                           "source": "report v1 R4 (this machine)" if arch == "MLP01"
                           else "2026-11-02 architecture ablation (this machine)"})
    for arch in ["MLP02", "MLP04", "MLP06"]:
        naming.append({"report_name": "OU + Gate (f-DAgger round 0)", "size": size_label(arch), "row": "R4",
                       "round": 0, "seed": 0, "run_id": canonical("R4", arch, 0, None, "v2", "2026-11-04"),
                       "original_run_id": tm[arch]["r0_run"], "source": "teammate (his machine), seed 0 only"})
    write_csv("naming.csv", naming)

    # ---------------------------------------------------------------- validation
    val = []
    for arch in ORDER:
        ou = [holds(OURS_R4[arch].format(s=s)) for s in seeds]
        if arch == "MLP01":
            fd = [holds(OURS_R8.format(s=s)) for s in seeds]
            r0, k = "", 3
        else:
            fd = [tm[arch]["seeds"][s] for s in seeds]
            r0, k = "%d/%d" % tm[arch]["r0_hold"], tm[arch]["round"]
        ou_r, fd_r = rate(sum(a for a, _ in ou), sum(b for _, b in ou)), rate(sum(a for a, _ in fd), sum(b for _, b in fd))
        val.append({"size": size_label(arch), "parameters": mlp_cost(SIZES[arch][0])[0],
                    "ou_gate_per_seed": " / ".join(str(a) for a, _ in ou), "ou_gate_pooled": fmt(ou_r),
                    "f_dagger_gate_round": k, "f_dagger_gate_per_seed": " / ".join(str(a) for a, _ in fd),
                    "f_dagger_gate_pooled": fmt(fd_r), "gain_pp": round(fd_r["pct"] - ou_r["pct"], 1),
                    "chain_round0_seed0": r0,
                    "chain_seed0_by_round": tm[arch]["by_round_s0"] if arch != "MLP01" else ""})
    write_csv("validation.csv", val)

    # ---------------------------------------------------------------- test
    test = []
    for arch in ORDER:
        counts = lt["R8"] if arch == "MLP01" else tm[arch]["test"]
        test.append({"method": "f-DAgger + Gate", "size": size_label(arch),
                     **{g: fmt(rate(*counts[g])) for g in GROUPS}})
    test.append({"method": "OU + Gate (reference)", "size": size_label("MLP01"),
                 **{g: fmt(rate(*lt["R4"][g])) for g in GROUPS}})
    test.append({"method": "Expert", "size": "-", **{g: fmt(rate(*lt["expert"][g])) for g in GROUPS}})
    same_expert = all(tuple(lt["expert"][g]) == tuple(tm["expert"][g]) for g in GROUPS)
    write_csv("test.csv", test)

    # ---------------------------------------------------------------- efficiency
    lat_local = {}
    p = DATA / "latency/latency_local.csv"
    if p.exists():
        for r in read(p):
            lat_local[r["architecture"].split("-")[0]] = r
    eff = []
    for arch in ORDER:
        params, flops = mlp_cost(SIZES[arch][0])
        row = {"size": size_label(arch), "parameters": params, "flops": flops,
               "params_vs_256x3": round(mlp_cost(SIZES["MLP01"][0])[0] / params, 1),
               "latency_p50_us_this_machine": lat_local.get(arch, {}).get("median_us", ""),
               "latency_p99_us_this_machine": lat_local.get(arch, {}).get("p99_us", ""),
               "npz_kb_this_machine": lat_local.get(arch, {}).get("npz_kb", "")}
        if arch == "MLP01":
            row.update({"latency_p50_us_teammate": "", "train_min_per_model_teammate": "",
                        "pipeline_min_all_rounds_teammate": "", "pipeline_min_to_chosen_teammate": ""})
        else:
            t = tm[arch]["top"]
            row.update({"latency_p50_us_teammate": t["latency_p50_us"], "train_min_per_model_teammate": t["train_min_median"],
                        "pipeline_min_all_rounds_teammate": t["pipeline_min_all_rounds"],
                        "pipeline_min_to_chosen_teammate": t["pipeline_min_to_chosen"]})
        eff.append(row)
    write_csv("efficiency.csv", eff)

    # ---------------------------------------------------------------- teammate models re-validated here
    rerun = []
    p = DATA / "validation/rerun_this_machine.json"
    if p.exists():
        local = json.loads(p.read_text())
        for arch in ["MLP02", "MLP04", "MLP06"]:
            for s in seeds:
                run = tm[arch]["runs"][s]
                if run in local:
                    n = local[run]["n"]
                    here = round(local[run]["sustained_hold_success"] * n)
                    rerun.append({"size": size_label(arch), "seed": s, "run_id": canonical("R8", arch, s, tm[arch]["round"], "v3", "2026-11-05"),
                                  "teammate_recorded": "%d/%d" % tm[arch]["seeds"][s], "this_machine": f"{here}/{n}",
                                  "median_final_err_mm_this_machine": round(local[run]["median_final_err_mm"], 2)})
        if rerun:
            write_csv("rerun.csv", rerun)

    # ---------------------------------------------------------------- LaTeX
    def pct(r):
        return f"{r['pct']:.1f}"
    lines = [r"\begin{table}[!htbp]",
             r"\caption{MLP Size: Validation and Test Success (\%) Across Hidden Widths.}",
             r"\label{tab:network-size}", r"\centering", r"\setlength{\tabcolsep}{2.5pt}",
             r"\begin{tabular}{@{}lrrrrrrrr@{}}", r"\toprule",
             r" & & \multicolumn{3}{c}{Validation} & \multicolumn{4}{c}{Test: f-DAgger+Gate} \\",
             r"\cmidrule(lr){3-5}\cmidrule(l){6-9}",
             r"Hidden & Params & \shortstack{OU\\+Gate} & \shortstack{f-DAgger\\+Gate} & $\Delta$ & \shortstack{T0+T4\\+T5} & "
             r"\shortstack{T2\\+T3} & T1 & All \\", r"\midrule"]
    for arch, v in zip(ORDER, val):
        counts = lt["R8"] if arch == "MLP01" else tm[arch]["test"]
        ou_k, fd_k = (sum(int(x) for x in v[c].split(" / ")) for c in ("ou_gate_per_seed", "f_dagger_gate_per_seed"))
        ou_r, fd_r = rate(ou_k, 96), rate(fd_k, 96)
        cells = [f"${SIZES[arch][0][0]}\\times3$", f"{v['parameters']:,}", pct(ou_r), pct(fd_r),
                 f"${fd_r['pct'] - ou_r['pct']:+.1f}$"] + [pct(rate(*counts[g])) for g in GROUPS]
        if arch == "MLP01":
            cells = [r"$\mathbf{256\times3}$"] + [r"$\mathbf{" + c.strip("$") + "}$" if c.startswith("$")
                                                  else r"\textbf{" + c + "}" for c in cells[1:]]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\midrule",
              "Expert & -- & -- & -- & -- & " + " & ".join(pct(rate(*lt["expert"][g])) for g in GROUPS) + r" \\",
              r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
              r"\begin{minipage}{\columnwidth}\footnotesize",
              r"Three hidden layers of width $w$. $\Delta$: f-DAgger+Gate minus OU+Gate on",
              r"validation (points). Student rates pool three seeds; the expert is",
              r"evaluated once. Validation: 32 scenes. Test: T0+T4+T5 112 scenes, T2+T3 56,",
              r"T1 72. Each width uses its own f-DAgger collection chain; $w\le128$ ran on a",
              r"second machine. Bold: the main model of Section~\ref{sec:performance}.",
              r"\end{minipage}", r"\end{table}"]
    (OUT / "network-size.tex").write_text("\n".join(lines) + "\n")

    # ---------------------------------------------------------------- Markdown
    def md(rows):
        keys = list(rows[0])
        return ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)] + \
               ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in rows]
    s = ["# Network size: summary tables", "",
         "Rebuilt by `code/make_tables.py`. Rates are % with 95% Wilson intervals and (successes/rollouts).", "",
         "## Validation (32 scenes, seeds 0-2 pooled)", ""] + md(val) + \
        ["", "## Unified test (240 scenes, seeds 0-2 pooled)", ""] + md(test) + \
        ["", f"Expert counts identical on both machines: **{same_expert}**.", "",
         "## Efficiency", ""] + md(eff)
    if rerun:
        s += ["", "## Teammate models re-validated on this machine", ""] + md(rerun)
    s += ["", "## Naming", ""] + md(naming)
    (OUT / "summary.md").write_text("\n".join(s) + "\n")
    print(f"wrote {len(list(OUT.iterdir()))} files to {OUT}; expert identical on both machines: {same_expert}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
