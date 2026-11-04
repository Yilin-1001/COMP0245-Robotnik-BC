"""Redraw figures/ou_noise_episode.png and figures/state_coverage.png with the report names.

Both figures come from the pipeline script evaluation/plot_pipeline_figures.py (functions `dart` and `state_coverage`),
which labels the OU-noise data "DART" (the run-id name). In the report, "DART" is full DART (Laskey et al., 2017) and
the phase-gated OU data are called "OU" (see ../../06_ablation/README.md). This wrapper runs the pipeline functions
unchanged and renames only the text of the finished figure before it is saved; the pipeline script is not edited.

Usage (from final/robotnik_arm_task/; needs data/Markovian/{Clean_253,DART} and data/datasets/A, git-ignored):
    python report/04_f_dagger/code/redraw_ou_figures.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text

TASK_DIR = Path(__file__).resolve().parents[3]  # final/robotnik_arm_task
FIG = Path(__file__).resolve().parents[1] / "figures"
sys.path.insert(0, str(TASK_DIR))
import evaluation.plot_pipeline_figures as ppf  # noqa: E402

RENAME = [  # applied in order; "DART-style" (the principle, after Laskey et al.) is kept
    ("Training-row coverage: DART adds perturbed states", "Training-row coverage: OU noise adds perturbed states"),
    ("DART rows show", "OU rows show"),
    ("DART (", "OU ("),
    ("DART draw", "OU draw"),
]
_finish = ppf.finish


def finish(fig, out, title, note=None):
    for a, b in RENAME:
        title = title.replace(a, b)
        note = note.replace(a, b) if note else note
    for t in fig.findobj(Text):
        s = t.get_text()
        for a, b in RENAME:
            s = s.replace(a, b)
        t.set_text(s)
    return _finish(fig, out, title, note)


if __name__ == "__main__":
    ppf.finish = finish
    ppf.style(plt)
    for name, fn in (("ou_noise_episode", ppf.dart), ("state_coverage", ppf.state_coverage)):
        print(fn(plt, FIG / f"{name}.png"))
        plt.close("all")
