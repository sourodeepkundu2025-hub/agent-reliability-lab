"""Paired task-cluster bootstrap: seeds are repeats, tasks are sampling units."""
import random
import statistics


def paired_summary(rows: list[dict], baseline: str, candidate: str, repetitions=2000) -> dict:
    pairs = {}
    for row in rows:
        if row["policy"] in {baseline, candidate}:
            pairs.setdefault((row["task_id"], row["seed"]), {})[row["policy"]] = int(row["success"])
    clusters = {}
    for (task_id, _), pair in pairs.items():
        if set(pair) != {baseline, candidate}:
            raise ValueError("unpaired observation")
        clusters.setdefault(task_id, []).append(pair[candidate] - pair[baseline])
    values = [statistics.mean(value) for value in clusters.values()]
    if not values:
        raise ValueError("no paired tasks")
    rng = random.Random(20261020)
    draws = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(repetitions))
    return {"baseline": baseline, "candidate": candidate, "success_delta": statistics.mean(values),
            "ci95_percentile": [draws[int(.025 * repetitions)], draws[int(.975 * repetitions)]],
            "task_units": len(values), "paired_task_seed_runs": len(pairs),
            "method": "paired task-cluster bootstrap; average seeds within task; 2000 resamples",
            "interpretation": "conditional synthetic task-family uncertainty; not real-agent evidence"}
