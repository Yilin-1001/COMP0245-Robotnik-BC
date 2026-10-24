"""Frozen scenario sets. Seeds here never overlap with training seeds.

val: validation, used for model selection
T1:  in-distribution test
T2:  structured grid, 8 directions x 3 distances x 3 heights, nominal physics (72)
T3:  robustness, friction x0.5/x1.5, masses +-25%, double sensor noise,
     mid-episode push, targets up to 3.5 m, a few unreachable targets (100)
"""


def get_split(name: str) -> list:
    raise NotImplementedError
