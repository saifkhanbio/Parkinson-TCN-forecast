"""Origin-restricted population controls and coherent burden transformations."""
import itertools

import numpy as np
import pandas as pd


def population_forecast(panel, config, origin, target, outcome, method="log_trend_last8"):
    if method not in {"log_trend_last8", "persistence"}:
        raise ValueError("Unknown operational population method")
    history = panel.loc[panel.location_name.eq(target) & panel.outcome.eq(outcome)
                        & panel.year.between(origin-7, origin)].copy()
    keys = ["sex", "age", "year"]
    expected = {(s, a, y) for s in config["sexes"] for a in config["ages"] for y in range(origin-7, origin+1)}
    if len(history) != len(expected) or set(map(tuple, history[keys].to_numpy())) != expected:
        raise ValueError("Population history requires eight complete years")
    values = history[["rate", "count"]].to_numpy()
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Population inference requires positive finite rates and counts")
    history["population"] = history["count"] / history.rate * 100000
    records = []
    for (sex, age), part in history.groupby(["sex", "age"]):
        part = part.sort_values("year")
        logs = np.log(part.population.to_numpy())
        if method == "log_trend_last8":
            coefficients = np.linalg.lstsq(np.column_stack([np.ones(8), np.arange(-7, 1)]), logs, rcond=None)[0]
        else:
            coefficients = [logs[-1], 0.]
        for horizon in config["calendar"]["horizons"]:
            log_population = coefficients[0] + coefficients[1]*horizon
            status, reason = "ok", ""
            if not np.isfinite(log_population) or abs(log_population) > 700:
                log_population, status, reason = logs[-1], "fallback", "Unsafe population forecast"
            records.append({"target": target, "outcome": outcome, "origin": origin, "sex": sex, "age": age,
                            "horizon": horizon, "forecast_year": origin+horizon, "population_method": method,
                            "log_population": float(log_population), "population": float(np.exp(log_population)),
                            "population_status": status, "population_fallback_reason": reason,
                            "last_population_input_year": origin})
    return pd.DataFrame(records)


def population_residuals(panel, config, fit_origin, target, outcome, method, residual_origins):
    """Joint population log errors, centered on the SAME origins as rate errors."""
    if len(set(residual_origins)) != len(residual_origins) or not residual_origins:
        raise ValueError("Need distinct historical population residual origins")
    if any(o + max(config["calendar"]["horizons"]) > fit_origin for o in residual_origins):
        raise ValueError("Uncompleted population residual block")
    allowed = panel.loc[panel.year.le(fit_origin)].copy()
    truth = allowed.loc[allowed.location_name.eq(target) & allowed.outcome.eq(outcome)].copy()
    truth["actual_population"] = truth["count"] / truth.rate * 100000
    truth = truth.rename(columns={"year": "forecast_year"})
    frames = []
    for origin in residual_origins:
        forecast = population_forecast(allowed, config, origin, target, outcome, method)
        frame = forecast.merge(truth[["sex", "age", "forecast_year", "actual_population"]],
                               on=["sex", "age", "forecast_year"], how="left", validate="many_to_one")
        if not np.isfinite(frame.actual_population).all() or not frame.actual_population.gt(0).all():
            raise ValueError("Invalid or missing population verification")
        frame["raw_population_log_error"] = np.log(frame.actual_population) - frame.log_population
        frame["residual_origin"] = origin
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result["mean_population_log_error"] = result.groupby(["sex", "age", "horizon"]).raw_population_log_error.transform("mean")
    result["centered_population_log_error"] = result.raw_population_log_error - result.mean_population_log_error
    result["fit_origin"] = fit_origin
    return result


def hierarchy(config):
    """31-node hierarchy: 22 cells, six broad-age/sex, two sex totals, grand total."""
    bottom = [(sex, age) for sex in config["sexes"] for age in config["ages"]]
    rows, matrices = [], []
    for i, (sex, age) in enumerate(bottom):
        rows.append({"node": f"{sex}__{age}", "level": "age_sex", "sex": sex, "age_group": age})
        matrices.append(np.eye(len(bottom))[i])
    for sex in config["sexes"]:
        for group, starts in config["age_groups"].items():
            rows.append({"node": f"{sex}__{group}", "level": "broad_age_sex", "sex": sex, "age_group": group})
            matrices.append(np.array([float(s == sex and int(a.split('-')[0].rstrip('+')) in starts) for s, a in bottom]))
    for sex in config["sexes"]:
        rows.append({"node": f"{sex}__45+", "level": "sex_total", "sex": sex, "age_group": "45+"})
        matrices.append(np.array([float(s == sex) for s, _ in bottom]))
    rows.append({"node": "Both__45+", "level": "grand_total", "sex": "Both", "age_group": "45+"})
    matrices.append(np.ones(len(bottom)))
    return np.stack(matrices), pd.DataFrame(rows), bottom


def count_and_share_views(frame, config, value="count", identifiers=None):
    """Aggregate complete sex/age cells; shared draw IDs remain in identifiers."""
    identifiers = identifiers or ["target", "outcome", "origin", "family", "horizon", "forecast_year", "population_method"]
    matrix, nodes, bottom = hierarchy(config)
    expected = set(bottom)
    if not np.isfinite(frame[value]).all() or not frame[value].gt(0).all():
        raise ValueError("Positive finite cell counts required")
    counts, shares = [], []
    for key, part in frame.groupby(identifiers, sort=False, dropna=False):
        if len(part) != len(bottom) or set(zip(part.sex, part.age)) != expected:
            raise ValueError("Count aggregation needs complete unique age/sex cells")
        context = dict(zip(identifiers, key if isinstance(key, tuple) else [key]))
        ordered = part.set_index(["sex", "age"]).reindex(pd.MultiIndex.from_tuples(bottom))[value].to_numpy()
        table = nodes.copy()
        table[value] = matrix @ ordered
        for name, item in context.items():
            table[name] = item
        counts.append(table)
        for sex in config["sexes"]:
            sex_values = np.array([v for (s, _), v in zip(bottom, ordered) if s == sex])
            ages = np.array([int(age.split('-')[0].rstrip('+')) for age in config["ages"]])
            for threshold in [65, 80]:
                shares.append({**context, "sex": sex, "threshold": threshold,
                               "share": float(sex_values[ages >= threshold].sum()/sex_values.sum())})
    return pd.concat(counts, ignore_index=True), pd.DataFrame(shares)


def rate_ratio_views(frame, config, value="prediction", identifiers=None):
    identifiers = identifiers or ["target", "outcome", "origin", "family", "horizon", "forecast_year", "age"]
    if frame.duplicated(identifiers + ["sex"]).any():
        raise ValueError("Duplicate sex-ratio cells")
    pivot = frame.pivot(index=identifiers, columns="sex", values=value)
    if set(pivot.columns) != set(config["sexes"]) or not np.isfinite(pivot.to_numpy()).all() or (pivot <= 0).any().any():
        raise ValueError("Sex ratios require positive paired male and female rates")
    pivot["male_female_rate_ratio"] = pivot["Male"] / pivot["Female"]
    return pivot.reset_index()[identifiers + ["male_female_rate_ratio"]]


def joint_count_draws(rate_draws, population, population_errors):
    keys = ["target", "outcome", "origin", "sex", "age", "horizon", "forecast_year"]
    joined = rate_draws.merge(population[keys + ["population_method", "log_population"]],
                              on=keys, how="left", validate="many_to_one")
    # The historical origin is retained separately as residual_origin.
    errors = population_errors.drop(columns="origin").rename(columns={"fit_origin": "origin"})
    error_keys = ["target", "outcome", "origin", "sex", "age", "horizon", "residual_origin", "population_method"]
    joined = joined.merge(errors[error_keys + ["centered_population_log_error"]],
                            on=error_keys, how="left", validate="many_to_one")
    fields = ["log_population", "centered_population_log_error", "log_draw"]
    if not np.isfinite(joined[fields]).all().all():
        raise ValueError("Rate and population draws must share complete historical blocks")
    joined["population_draw"] = np.exp(joined.log_population + joined.centered_population_log_error)
    joined["count"] = np.exp(joined.log_draw) * joined.population_draw / 100000
    return joined


def shapley_change(start_population, start_rate, end_population, end_rate):
    """Exact symmetric accounting for population size, composition and rates."""
    inputs = [np.asarray(x, dtype=float) for x in [start_population, start_rate, end_population, end_rate]]
    if len({x.shape for x in inputs}) != 1 or any(x.ndim != 1 or not np.isfinite(x).all() or (x <= 0).any() for x in inputs):
        raise ValueError("Positive matching age vectors required for decomposition")
    n0, r0, n1, r1 = inputs
    states = [(n0.sum(), n0/n0.sum(), r0), (n1.sum(), n1/n1.sum(), r1)]
    def burden(indices):
        size, composition, rates = [states[indices[i]][i] for i in range(3)]
        return float(size * np.dot(composition, rates) / 100000)
    contributions = np.zeros(3)
    for order in itertools.permutations(range(3)):
        active = [0, 0, 0]
        previous = burden(active)
        for item in order:
            active[item] = 1
            current = burden(active)
            contributions[item] += (current-previous)/6
            previous = current
    change = burden([1, 1, 1]) - burden([0, 0, 0])
    np.testing.assert_allclose(contributions.sum(), change, atol=1e-10, rtol=1e-12)
    return dict(zip(["population_size", "age_composition", "rate_component"], map(float, contributions))), change
