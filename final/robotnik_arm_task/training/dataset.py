"""Load .npz datasets, split by scenario seed (never by timestep), normalise.

Normalisation statistics come from the training split only.
"""


def load_datasets(paths: list[str]):
    raise NotImplementedError


def split_by_seed(data, val_fraction: float, seed: int):
    raise NotImplementedError


def fit_normaliser(train_data):
    raise NotImplementedError
