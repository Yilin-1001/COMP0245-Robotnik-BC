"""Archived on 2026-11-01 from training/common/dataset.py: helpers used only by the archived clean-pipeline prepare_run (engine_legacy.py).

Kept for reference only; it is not importable as is (its imports refer to the original module)."""

def select_refs(refs: list[EpisodeRef], ids: list[str]) -> list[EpisodeRef]:
    by_id = {r.scenario_id: r for r in refs}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise ValueError(f"scenarios missing from the source inventory: {missing[:5]}")
    return [by_id[i] for i in ids]


def cache_id(refs: list[EpisodeRef], builder_id: str) -> str:
    key = json.dumps({"sources": [[r.scenario_id, r.sha256] for r in refs], "builder": builder_id,
                      "labels": LABEL_NAMES}, sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def subset_cache(cache: EpisodeCache, ids: list[str]) -> EpisodeCache:
    """Episodes of an existing cache, in the given order (used for e.g. the one-episode sanity run)."""
    pos = {e: i for i, e in enumerate(cache.episode_ids)}
    rows = [np.arange(cache.offsets[pos[i]], cache.offsets[pos[i] + 1]) for i in ids]
    idx = np.concatenate(rows)
    offsets = np.concatenate(([0], np.cumsum([len(r) for r in rows]))).astype(np.int64)
    meta = dict(cache.metadata, episode_ids=list(ids),
                source_sha256=[cache.metadata["source_sha256"][pos[i]] for i in ids], n_rows=int(len(idx)))
    return EpisodeCache(cache.x[idx], cache.y[idx], offsets, list(ids),
                        {k: v[idx] for k, v in cache.diagnostics.items()}, meta)
