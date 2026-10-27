# Frozen experiment protocols

Each `p-<hash>.yaml` file records reviewed numerical decisions for the clean-data MLP experiments, with the
SHA-256 of the evidence each decision rests on (for example the pilot `metrics.csv` curves or rollout summaries).

- Only `python -m experiments.run --freeze ...` writes protocols. The ID is the hash of the decisions, evidence and
  parent ID, so a file edited in place fails validation when it is loaded.
- A protocol is never rewritten. Changing or adding a decision creates a new file whose `parent` is the previous ID;
  an already frozen decision cannot be changed in a child.
- Protocols select explicit launcher values (`training.updates`, `training.lr`, `loss.action`,
  `loss.jacobian.calibration`, ...). They are not a hidden configuration layer: every run's resolved config and
  identity are saved in its own directory.
- Raw curves, checkpoints and rollout details stay local; protocols are small and versioned.

Decision groups: `budget` (updates, pilot_lr, plateau evidence), `loss` (action), `lr` (per teacher/architecture
from the offline proxy), `finalists`, `selected`, `calibration` (per teacher λ_J record), `final` (frozen test
protocol, Task 11) and `notes`.
