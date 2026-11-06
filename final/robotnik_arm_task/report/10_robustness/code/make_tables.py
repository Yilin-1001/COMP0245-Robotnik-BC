"""Robustness to unseen physics changes, sensor noise and pushes: rebuild ../tables from ../data/run.

Run: evaluation.robust_probe on the report's final models (checkpoints/2026-11-05final): R4 OU + Gate (v1),
R5-R8 DAgger / DAgger + Gate / f-DAgger / f-DAgger + Gate (v3, round 3), seeds 0-2, plus teacher A once.
12 in-range scenes per condition (development seeds 990000+, disjoint from every frozen split), 9 conditions,
108 scenes shared by all 16 policies. Full DART (R9) was not run.

Tables:
  rates.csv         sustained hold per condition and method, 3 seeds pooled (n = 36; expert n = 12), 95% Wilson
  expert_held.csv   the same, restricted to scenes the expert holds under that condition
  seeds.csv         hold count per seed
  final_error.csv   median final error (mm) over all rollouts of each condition and method
  robustness.tex    LaTeX table for the paper (Results, Robustness Performance)
  summary.md        all of the above as Markdown

Success is robust_probe's predicate: reached below 10 mm, at least 8 s elapsed, and every step of the final 1 s of a
3 s post-reach window below 10 mm; the at-target pushes use a 10 s window so the push lands inside it. This is NOT the
unified test's 3 s continuous hold. Wilson intervals treat the 36 pooled rollouts as independent; the three seeds
share the 12 scenes, so they understate the uncertainty. No simulation runs here.

Usage (from final/robotnik_arm_task/):
    python report/10_robustness/code/make_tables.py
"""
import csv
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
RUN, OUT = HERE / "data/run", HERE / "tables"
# row prefix -> (report name, method, gate)
METHODS = {"teacher-A": ("Expert", "Expert", "--"), "R4": ("OU + Gate", "OU", "Yes"), "R5": ("DAgger", "DAgger", "No"),
           "R6": ("DAgger + Gate", "DAgger", "Yes"), "R7": ("f-DAgger", "f-DAgger", "No"),
           "R8": ("f-DAgger + Gate", "f-DAgger", "Yes")}
LABELS = {"nominal": "Nominal", "friction x0.5": r"Friction $\times0.5$", "friction x1.5": r"Friction $\times1.5$",
          "mass x0.9": r"Mass $\times0.9$", "mass x1.1": r"Mass $\times1.1$",
          "sensor noise x20": "Sensor noise", "push base, driving": "Base push, driving",
          "push base, at target": "Base push, at target", "push hand, at target": "Hand push, at target"}


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def fmt(k: int, n: int) -> str:
    if n == 0:
        return "-"
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.1f} [{lo:.1f}, {hi:.1f}] ({k}/{n})"


def method_of(policy: str) -> str:
    return "teacher-A" if policy == "teacher-A" else policy[:2]


def write_csv(name: str, rows: list[dict]) -> None:
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def md(rows: list[dict]) -> list[str]:
    keys = list(rows[0])
    return ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)] + \
           ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in rows]


def main() -> int:
    OUT.mkdir(exist_ok=True)
    rows = list(csv.DictReader(open(RUN / "rollouts.csv")))
    for r in rows:
        r["method"], r["ok"] = method_of(r["policy"]), r["sustained_hold_pass"] == "True"
    conditions = list(dict.fromkeys(r["condition"] for r in rows))
    held = {(r["condition"], r["scenario_id"]) for r in rows if r["method"] == "teacher-A" and r["ok"]}

    def sel(cond, m, only_held=False):
        return [r for r in rows if r["condition"] == cond and r["method"] == m
                and (not only_held or (cond, r["scenario_id"]) in held)]

    rates, kept, seeds, errors, counts = [], [], [], [], {}
    for c in conditions:
        line, line_h, line_s, line_e = {"condition": c}, {"condition": c}, {"condition": c}, {"condition": c}
        for m, (name, _, _) in METHODS.items():
            rs, rh = sel(c, m), sel(c, m, True)
            k, kh = sum(r["ok"] for r in rs), sum(r["ok"] for r in rh)
            counts[(c, m)] = (k, len(rs))
            line[name], line_h[name] = fmt(k, len(rs)), fmt(kh, len(rh))
            by_seed = {}
            for r in rs:
                by_seed.setdefault(r["policy"], []).append(r["ok"])
            line_s[name] = " / ".join(f"{sum(v)}/{len(v)}" for _, v in sorted(by_seed.items()))
            line_e[name] = round(statistics.median(float(r["final_err_mm"]) for r in rs), 2)
        rates.append(line), kept.append(line_h), seeds.append(line_s), errors.append(line_e)
    perturbed = [c for c in conditions if c != "nominal"]
    for label, group in (("all 8 perturbations", perturbed), ("all 9 conditions", conditions)):
        line, line_h = {"condition": label}, {"condition": label}
        for m, (name, _, _) in METHODS.items():
            rs = [r for c in group for r in sel(c, m)]
            rh = [r for c in group for r in sel(c, m, True)]
            counts[(label, m)] = (sum(r["ok"] for r in rs), len(rs))
            line[name] = fmt(*counts[(label, m)])
            line_h[name] = fmt(sum(r["ok"] for r in rh), len(rh))
            counts[(label + " | expert-held", m)] = (sum(r["ok"] for r in rh), len(rh))
        rates.append(line), kept.append(line_h)
    write_csv("rates.csv", rates)
    write_csv("expert_held.csv", kept)
    write_csv("seeds.csv", seeds)
    write_csv("final_error.csv", errors)

    # ---------------------------------------------------------------- LaTeX (expert-robust conditions first)
    students = [m for m in METHODS if m != "teacher-A"]
    expert_ok = [c for c in conditions if counts[(c, "teacher-A")][0] == counts[(c, "teacher-A")][1]]
    expert_fail = [c for c in conditions if c not in expert_ok]

    def line_tex(c):
        k, n = counts[(c, "teacher-A")]
        cells = [LABELS.get(c, c), f"{100 * k / n:.1f}"]
        vals = [100 * counts[(c, m)][0] / counts[(c, m)][1] for m in students]
        best = max(vals)
        cells += [(r"\textbf{%.1f}" if v == best else "%.1f") % v for v in vals]
        return " & ".join(cells) + r" \\"

    lines = [r"\begin{table}[!htbp]",
             r"\caption{Robustness: Hold Success (\%) Under Unseen Perturbations.}",
             r"\label{tab:robustness}", r"\centering", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r"Condition & Expert & OU & DAgger & DAgger & f-DAgger & f-DAgger \\",
             r"Gate & -- & Yes & No & Yes & No & Yes \\", r"\midrule",
             r"\multicolumn{7}{@{}l}{\textit{Expert holds every scene}} \\"]
    lines += [line_tex(c) for c in expert_ok]
    lines += [r"\midrule", r"\multicolumn{7}{@{}l}{\textit{Expert fails in some scenes}} \\"]
    lines += [line_tex(c) for c in expert_fail]
    LABELS["all 8 perturbations"] = "All perturbations"
    # Same eight perturbations, restricted to the scenes the expert holds.
    LABELS["all 8 perturbations | expert-held"] = "Expert-held scenes"
    lines += [r"\midrule", line_tex("all 8 perturbations"), line_tex("all 8 perturbations | expert-held")]
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip", r"\begin{minipage}{\columnwidth}\footnotesize",
              r"12 in-range scenes per condition; students pool three seeds ($n=36$), the expert",
              r"runs once. Success: final 1\,s below 10\,mm after reaching. Pushes last 0.2\,s:",
              r"120\,N on the base, 20\,N on the hand, at target 1\,s after first reaching.",
              r"Bottom rows pool the eight perturbations (96 scenes); expert-held keeps the 76",
              r"perturbed scenes the expert holds. Bold: best student per row.",
              r"\end{minipage}", r"\end{table}"]
    (OUT / "robustness.tex").write_text("\n".join(lines) + "\n")

    verification = json.loads((RUN / "verification.json").read_text()) if (RUN / "verification.json").exists() else {}
    s = ["# Robustness: summary tables", "",
         "Rebuilt by `code/make_tables.py`. Rates are % with 95% Wilson intervals and (holds/rollouts).",
         f"Run verification (`code/verify_run.py`): all checks passed = **{verification.get('all_ok')}**.", "",
         "## Sustained hold, all scenes", ""] + md(rates) + \
        ["", "## Sustained hold, scenes the expert holds", ""] + md(kept) + \
        ["", "## Holds per seed (s0 / s1 / s2)", ""] + md(seeds) + \
        ["", "## Median final error (mm)", ""] + md(errors)
    (OUT / "summary.md").write_text("\n".join(s) + "\n")
    print(f"wrote {len(list(OUT.iterdir()))} files to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
