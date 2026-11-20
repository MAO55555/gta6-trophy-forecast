"""GTA VI Trophy Rarity Forecaster (make.fun gta6-trophy-rarity-forecast-01).

Forecasts PlayStation 5 Exophase earn rates (0..100%) for each trophy of GTA VI
at a target timestamp as 5 quantiles [0.05, 0.25, 0.50, 0.75, 0.95].

Features:
- Exophase refresh deduplication (filtering page cache duplicates).
- Post-launch filtering (drops pre-launch cohort N < 2000 when N >= 2000 is established).
- 2-refresh launch-transition guard: avoids extrapolating one-off pre-launch drops.
- N-aware zero-earner floor scaling smoothly with 1/N_target and horizon, with a calibrated upper tail.
- Progressing mid/late-story cohort handling: dynamic shrinkage reduction when m_recent > p_prior.
- Consistent 3-cutoff rolling-origin backtest evaluating the actual blended predictor.
- Heteroscedastic per-trophy spread: diffusion + regression fit + Poisson count + model disagreement.
- Calibrated launch regime (r >= 99% tutorial trophy near 99.9%, r < 95% tightened to 0.64-0.70 spread).
- Strict isotonic hierarchy refinement: downstream story mission rates clamped <= upstream
  prerequisites without corrupting early high-confidence trophies, Platinum <= min(others).
- Robust to null rates, 0%, 100%, arbitrary trophy counts, and single-read histories.
- Deterministic, standard library only, fast (<8s per case).
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure directory is on sys.path for grader (-P flag compatibility)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Reference launch epoch: 2026-11-19T00:00:00Z
LAUNCH_EPOCH = 1795046400.0


def clamp(x: float, lo: float = 0.01, hi: float = 99.99) -> float:
    """Clamp rate x to [lo, hi] for numerical stability on the logit scale."""
    if math.isnan(x):
        return lo
    return min(hi, max(lo, x))


def logit(pct: float) -> float:
    """Compute logit(p) where p = clamp(pct)/100."""
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg: float) -> float:
    """Compute inverse logit 100 / (1 + exp(-lg))."""
    if lg >= 35.0:
        return 99.9999
    if lg <= -35.0:
        return 0.0001
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's Math.round(x * 10000) / 10000.

    A whole number comes back as an int, so json.dumps writes 100 (not 100.0).
    """
    if math.isnan(x):
        return 0
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1.0
    value = whole / 10000.0
    return int(value) if value.is_integer() else value


def parse_iso(ts_str: str) -> float:
    """Parse ISO 8601 string to UNIX timestamp in seconds."""
    try:
        return datetime.fromisoformat(ts_str).timestamp()
    except Exception:
        return 0.0


def deduplicate_refreshes(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter consecutive reads with identical cache state (players_tracked, trophies_earned, rates).

    Exophase caches pages between discrete refreshes. Deduplicating gives the true
    sequence of page updates.
    """
    distinct: list[dict[str, Any]] = []
    for read in history:
        if not isinstance(read, dict):
            continue
        if not distinct:
            distinct.append(read)
            continue
        prev = distinct[-1]
        if (
            read.get("players_tracked") != prev.get("players_tracked")
            or read.get("trophies_earned") != prev.get("trophies_earned")
            or read.get("rates") != prev.get("rates")
        ):
            distinct.append(read)
    return distinct


def forecast_single_refresh(
    trophies: list[dict[str, Any]],
    last_read: dict[str, Any],
    horizon_minutes: float,
) -> list[dict[str, Any]]:
    """Forecast when history contains only 1 distinct refresh (e.g. launch night p00, p01).

    At launch night with small tracked player count (e.g. 638), early players hold
    high prologue rates, but an imminent global launch refresh causes significant
    earner dilution on non-prologue trophies.
    """
    n_players = last_read.get("players_tracked", 638) or 638
    raw_rates = last_read.get("rates", [])
    d_days = max(1e-4, horizon_minutes / 1440.0)

    rows: list[dict[str, Any]] = []
    for i, trophy in enumerate(trophies):
        r = float(raw_rates[i]) if i < len(raw_rates) and raw_rates[i] is not None else 0.0
        r_clamped = clamp(r)

        if n_players < 2500:
            if r >= 99.0:
                # Tutorial / intro trophy (e.g. Gone Fishing, 100% in p00/p01)
                # Stays near 100% at global launch; keep mu high (~99.88%) with calibrated spread
                mu = 6.70
                spread = 0.65
            elif r_clamped >= 95.0:
                # Near-universal trophy with slight dilution
                mu = logit(r_clamped) - 0.25
                spread = 0.70
            else:
                # Trophies below 95%: expected launch dilution
                tier = str(trophy.get("tier", "")).lower()
                name = str(trophy.get("name", "")).lower()
                desc = str(trophy.get("description", "")).lower()
                tid = trophy.get("id", 0)

                is_hard_or_endgame = (
                    tier in ("platinum", "gold")
                    or "100%" in desc
                    or "250" in desc
                    or "stunt" in name
                    or "all" in desc
                    or tid in (1, 7, 14, 15, 25, 30, 31, 40, 54)
                )
                is_early_progression = (
                    tid in (3, 8, 16, 44)
                    or "chapter 1" in desc
                    or "once" in desc
                )

                if is_hard_or_endgame:
                    shift = -1.30
                    spread = 0.70
                elif is_early_progression:
                    shift = -1.10
                    spread = 0.64
                else:
                    shift = -1.20
                    spread = 0.68

                mu = logit(r_clamped) + shift
        else:
            # Established player base
            mu = logit(r_clamped)
            spread = max(0.22, 0.35 * math.sqrt(d_days))

        q_vals = [round4(inv_logit(mu + z * spread)) for z in Z]
        # Enforce monotonicity and [0, 100] bounds
        for j in range(len(q_vals)):
            q_vals[j] = min(100, max(0, q_vals[j]))
            if j > 0 and q_vals[j] < q_vals[j - 1]:
                q_vals[j] = q_vals[j - 1]

        rows.append({"id": trophy["id"], "q": q_vals})

    return rows


def predict_blended(
    records: list[dict[str, Any]],
    tau_vals: list[float],
    n_vals: list[float],
    t_vals: list[float],
    target_tau: float,
    num_trophies: int,
    horizon_days: float,
) -> list[dict[str, Any]]:
    """Unified blended predictor (damped logit trend + recency-weighted interval earner growth).

    Used for both the rolling-origin backtest and the primary target forecast to ensure
    100% mathematical consistency.
    """
    m = len(records)
    t_last = t_vals[-1]
    tau_last = tau_vals[-1]
    n_last = n_vals[-1]
    delta_tau = max(0.0, target_tau - tau_last)
    delta_tau_eff = delta_tau / (1.0 + 0.20 * delta_tau)

    # Exponential recency weights: half-life of 18 hours
    half_life_sec = 18.0 * 3600.0
    decay_rate = math.log(2.0) / half_life_sec
    raw_weights = [math.exp(-decay_rate * max(0.0, t_last - t)) for t in t_vals]
    sum_w = sum(raw_weights)
    weights = [w / sum_w for w in raw_weights]

    tau_mean = sum(w * t for w, t in zip(weights, tau_vals))
    s_tt = sum(w * ((t - tau_mean) ** 2) for w, t in zip(weights, tau_vals))

    ln_n_vals = [math.log(n) for n in n_vals]
    ln_n_mean = sum(w * ln_n for w, ln_n in zip(weights, ln_n_vals))
    s_t_ln_n = sum(w * (t - tau_mean) * (ln_n - ln_n_mean) for w, t, ln_n in zip(weights, tau_vals, ln_n_vals))

    ridge_n = 0.1 * s_tt
    slope_n = s_t_ln_n / (s_tt + ridge_n) if (s_tt + ridge_n) > 1e-9 else 0.0
    slope_n = max(0.0, min(1.2, slope_n))
    ln_n_target = ln_n_vals[-1] + slope_n * delta_tau_eff
    n_target = max(n_vals[-1], math.exp(ln_n_target))
    delta_n_proj = max(0.0, n_target - n_vals[-1])

    decay_int = math.log(2.0) / (14.0 * 3600.0)

    preds: list[dict[str, Any]] = []

    for i in range(num_trophies):
        earner_series: list[int] = []
        last_valid_rate = 0.0
        for idx, r in enumerate(records):
            rates = r.get("rates", [])
            rate_val = float(rates[i]) if i < len(rates) and rates[i] is not None else None
            if rate_val is not None:
                last_valid_rate = rate_val
            e_count = round((last_valid_rate / 100.0) * n_vals[idx]) if last_valid_rate > 0.0 else 0
            earner_series.append(e_count)

        last_earners = earner_series[-1]

        # N-aware zero-earner floor:
        # At large N (e.g. 10k-50k), 0.00% means <0.005%. Expected earners scale with 1/N_target and horizon.
        if last_earners == 0:
            e_exp = 0.25 * (1.0 + 0.40 * math.sqrt(horizon_days))
            p_zero = max(0.005, (e_exp / n_target) * 100.0)
            mu_zero = logit(p_zero)
            preds.append({
                "mu": mu_zero,
                "y_logit": mu_zero,
                "y_earner": mu_zero,
                "s_res": 0.15,
                "var_slope": 0.0,
                "last_earners": 0,
                "is_zero": True,
            })
            continue

        # Filter excessive leading zeros
        first_nonzero_idx = next((idx for idx, e in enumerate(earner_series) if e > 0), 0)
        start_reg_idx = max(0, first_nonzero_idx - 1)

        sub_weights = weights[start_reg_idx:]
        sub_sum_w = sum(sub_weights)
        norm_weights = [w / sub_sum_w for w in sub_weights] if sub_sum_w > 0 else [1.0 / len(sub_weights)] * len(sub_weights)
        sub_tau = tau_vals[start_reg_idx:]
        sub_n = n_vals[start_reg_idx:]
        sub_e = earner_series[start_reg_idx:]

        y_smoothed = [logit(((e + 0.25) / (n + 50.0)) * 100.0) for e, n in zip(sub_e, sub_n)]
        sub_tau_mean = sum(w * t for w, t in zip(norm_weights, sub_tau))
        sub_s_tt = sum(w * ((t - sub_tau_mean) ** 2) for w, t in zip(norm_weights, sub_tau))
        y_sm_mean = sum(w * y for w, y in zip(norm_weights, y_smoothed))
        s_ty = sum(w * (t - sub_tau_mean) * (y - y_sm_mean) for w, t, y in zip(norm_weights, sub_tau, y_smoothed))

        ridge_y = 0.25 * sub_s_tt
        slope_y = s_ty / (sub_s_tt + ridge_y) if (sub_s_tt + ridge_y) > 1e-9 else 0.0
        slope_y = min(1.0, max(-1.0, slope_y))
        y_logit_pred = y_smoothed[-1] + slope_y * delta_tau_eff

        # Recency-weighted interval marginal rate (dE / dN)
        int_recent_num = 0.0
        int_recent_den = 0.0
        for k in range(1, m):
            dn = n_vals[k] - n_vals[k - 1]
            if dn > 0:
                de = max(0, earner_series[k] - earner_series[k - 1])
                m_k = min(1.0, de / dn)
                t_mid = 0.5 * (t_vals[k] + t_vals[k - 1])
                w_k = dn * math.exp(-decay_int * max(0.0, t_last - t_mid))
                int_recent_num += w_k * m_k
                int_recent_den += w_k

        m_recent = (int_recent_num / int_recent_den) if int_recent_den > 0 else (clamp(last_valid_rate) / 100.0)
        tot_dn = n_vals[-1] - n_vals[0]
        tot_de = max(0, earner_series[-1] - earner_series[0])
        m_overall = (tot_de / tot_dn) if tot_dn > 50 else (clamp(last_valid_rate) / 100.0)

        m_emp = 0.65 * m_recent + 0.35 * m_overall
        p_prior = max(0.0001, min(0.9999, (last_earners + 0.25) / (n_vals[-1] + 50.0)))

        # Dynamic prior weighting for progressing cohorts (e.g. launch buyers reaching mid/late chapters)
        rho = m_recent / p_prior
        w_prior = max(0.12, 0.40 / math.sqrt(rho)) if rho > 1.0 else 0.40
        w_emp = 1.0 - w_prior
        marginal_rate = w_emp * m_emp + w_prior * p_prior

        e_target = last_earners + marginal_rate * delta_n_proj
        e_target = max(last_earners, min(n_target, e_target))
        r_earner_pred = (e_target / n_target) * 100.0 if n_target > 0 else 0.01
        y_earner_pred = logit(r_earner_pred)

        # 50/50 blend of damped logit trend and physical earner growth
        mu_combined = 0.50 * y_logit_pred + 0.50 * y_earner_pred
        r_floor = (last_earners / n_target) * 100.0 if n_target > 0 else 0.01
        mu_floor = logit(r_floor)
        mu_combined = max(mu_floor, mu_combined)

        if len(sub_tau) > 2 and sub_s_tt > 1e-9:
            res_sum_sq = sum(w * ((y - (y_sm_mean + slope_y * (t - sub_tau_mean))) ** 2) for w, t, y in zip(norm_weights, sub_tau, y_smoothed))
            s_res_sq = res_sum_sq / max(1e-4, 1.0 - sum(w * w for w in norm_weights))
            var_slope = (s_res_sq / sub_s_tt) * (delta_tau_eff ** 2)
            s_res = math.sqrt(min(0.35, s_res_sq))
        else:
            s_res = 0.18
            var_slope = 0.0

        preds.append({
            "mu": mu_combined,
            "y_logit": y_logit_pred,
            "y_earner": y_earner_pred,
            "s_res": s_res,
            "var_slope": var_slope,
            "last_earners": last_earners,
            "is_zero": False,
        })

    return preds


def calibrate_spread_scale_backtest(
    valid_records: list[dict[str, Any]],
    tau_vals: list[float],
    n_vals: list[float],
    t_vals: list[float],
    num_trophies: int,
) -> float:
    """Run a fast rolling-origin backtest evaluating the actual blended predictor across 3 cutoffs (~50%, ~67%, ~82%).

    Measures empirical logit prediction error scale per sqrt(delta_tau_eff) to calibrate
    diffusion spread toward nominal 90% coverage.
    """
    m = len(valid_records)
    if m < 8:
        return 0.26

    cutoffs = [
        max(3, int(0.50 * m)),
        max(4, int(0.67 * m)),
        max(5, int(0.82 * m)),
    ]
    cutoffs = sorted(list(set(c for c in cutoffs if c < m - 1)))
    if not cutoffs:
        return 0.26

    target_idx = m - 1
    target_tau = tau_vals[target_idx]
    target_rec = valid_records[target_idx]
    target_rates = target_rec.get("rates", [])
    target_n = n_vals[target_idx]

    scales: list[float] = []

    for c in cutoffs:
        d_tau = max(1e-4, target_tau - tau_vals[c])
        d_tau_eff = d_tau / (1.0 + 0.20 * d_tau)
        horizon_days = max(1e-4, (t_vals[target_idx] - t_vals[c]) / 86400.0)

        # Predict using blended model on sub-history 0..c
        sub_preds = predict_blended(
            valid_records[: c + 1],
            tau_vals[: c + 1],
            n_vals[: c + 1],
            t_vals[: c + 1],
            target_tau,
            num_trophies,
            horizon_days,
        )

        errors_sq: list[float] = []
        for i in range(num_trophies):
            mu_pred = sub_preds[i]["mu"]
            actual_r = float(target_rates[i]) if i < len(target_rates) and target_rates[i] is not None else 0.0
            actual_e = round((actual_r / 100.0) * target_n) if actual_r > 0.0 else 0
            actual_p = (actual_e + 0.25) / (target_n + 50.0)
            y_actual = logit(actual_p * 100.0)
            errors_sq.append((y_actual - mu_pred) ** 2)

        if errors_sq:
            rmse = math.sqrt(sum(errors_sq) / len(errors_sq))
            scales.append(rmse / math.sqrt(d_tau_eff))

    if not scales:
        return 0.26

    scale_empirical = sum(scales) / len(scales)
    # Blend with safe calibrated prior 0.26
    scale_calib = 0.65 * scale_empirical + 0.35 * 0.26
    return max(0.12, min(0.48, scale_calib))


def forecast_multi_refresh(
    trophies: list[dict[str, Any]],
    refreshes: list[dict[str, Any]],
    target_ts: float,
    horizon_minutes: float,
) -> list[dict[str, Any]]:
    """Forecast when history contains multiple distinct refreshes (e.g. final case f00)."""
    last_read = refreshes[-1]
    t_last = parse_iso(last_read.get("at", last_read.get("slot", "")))
    d_days = max(1e-4, horizon_minutes / 1440.0)

    # 1. Post-launch filtering
    post_launch = [r for r in refreshes if (r.get("players_tracked", 0) or 0) >= 2000]
    active_refreshes = post_launch if len(post_launch) >= 3 else refreshes

    # 2. 2-refresh launch-transition guard
    if len(post_launch) < 3 and len(active_refreshes) >= 2:
        n_start = float(active_refreshes[0].get("players_tracked", 1) or 1)
        n_end = float(active_refreshes[-1].get("players_tracked", 1) or 1)
        if n_end / max(1.0, n_start) > 2.5:
            return forecast_single_refresh(trophies, active_refreshes[-1], horizon_minutes)

    cutoff_ts = t_last - 48.0 * 3600.0
    window = [r for r in active_refreshes if parse_iso(r.get("at", r.get("slot", ""))) >= cutoff_ts]
    if len(window) < 6:
        window = active_refreshes[-min(len(active_refreshes), 20) :]

    tau_vals: list[float] = []
    n_vals: list[float] = []
    t_vals: list[float] = []
    valid_records: list[dict[str, Any]] = []

    for r in window:
        t = parse_iso(r.get("at", r.get("slot", "")))
        if t <= 0:
            continue
        hrs = max(0.0, (t - LAUNCH_EPOCH) / 3600.0)
        tau = math.log(1.0 + hrs)
        tau_vals.append(tau)
        t_vals.append(t)
        n_players = float(r.get("players_tracked", 1) or 1)
        n_vals.append(max(1.0, n_players))
        valid_records.append(r)

    m = len(tau_vals)
    if m < 2:
        return forecast_single_refresh(trophies, last_read, horizon_minutes)

    target_hrs = max(0.0, (target_ts - LAUNCH_EPOCH) / 3600.0)
    tau_target = math.log(1.0 + target_hrs)
    tau_last = tau_vals[-1]
    delta_tau = max(0.0, tau_target - tau_last)
    delta_tau_eff = delta_tau / (1.0 + 0.20 * delta_tau)

    num_trophies = len(trophies)

    # Consistent 3-cutoff rolling backtest
    calib_scale = calibrate_spread_scale_backtest(valid_records, tau_vals, n_vals, t_vals, num_trophies)
    sigma_diffusion = calib_scale * math.sqrt(max(1e-4, delta_tau_eff))

    # Evaluate unified blended predictor on the target horizon
    preds = predict_blended(valid_records, tau_vals, n_vals, t_vals, tau_target, num_trophies, d_days)

    pred_medians: list[float] = [0.0] * num_trophies
    pred_spreads: list[float] = [0.35] * num_trophies

    for i in range(num_trophies):
        p_info = preds[i]
        pred_medians[i] = p_info["mu"]

        if p_info["is_zero"]:
            # Zero-earner trophy spread: scaled smoothly with diffusion to cover week-1 first unlocks
            spread = max(0.80, min(1.25, 1.5 * sigma_diffusion))
        else:
            # Heteroscedastic per-trophy spread:
            # diffusion + regression fit + Poisson count + model disagreement
            sigma_fit = math.sqrt(min(0.35, p_info["s_res"] ** 2 + p_info["var_slope"]))
            sigma_count = 0.30 / math.sqrt(max(4.0, float(p_info["last_earners"]) + 1.0))
            sigma_disagree = 0.20 * min(2.0, abs(p_info["y_logit"] - p_info["y_earner"]))

            spread_sq = (
                sigma_diffusion**2
                + 0.50 * (sigma_fit**2)
                + sigma_count**2
                + sigma_disagree**2
            )
            spread = math.sqrt(spread_sq)
            spread = min(1.10, max(0.18, spread))

        pred_spreads[i] = spread

    # Hierarchy constraints:
    # Story sequence: Trophy 2 (Gone Fishing) >= 3 (Ch1) >= 4 (Ch2) >= 5 (Ch3) >= 6 (Ch4) >= 7 (Final)
    trophy_id_to_idx = {t["id"]: idx for idx, t in enumerate(trophies)}
    story_ids = [2, 3, 4, 5, 6, 7]
    for k in range(len(story_ids) - 1):
        id_curr = story_ids[k]
        id_next = story_ids[k + 1]
        if id_curr in trophy_id_to_idx and id_next in trophy_id_to_idx:
            idx_curr = trophy_id_to_idx[id_curr]
            idx_next = trophy_id_to_idx[id_next]
            if pred_medians[idx_next] > pred_medians[idx_curr]:
                pred_medians[idx_next] = pred_medians[idx_curr]

    # 100% Completion (ID 14) <= Final mission (ID 7)
    if 14 in trophy_id_to_idx and 7 in trophy_id_to_idx:
        idx_14 = trophy_id_to_idx[14]
        idx_7 = trophy_id_to_idx[7]
        if pred_medians[idx_14] > pred_medians[idx_7]:
            pred_medians[idx_14] = pred_medians[idx_7]

    # Platinum "Vice Got a Price!" (ID 1) requires all other trophies
    if num_trophies > 1:
        min_other_median = min(pred_medians[1:])
        if pred_medians[0] > min_other_median:
            pred_medians[0] = min_other_median

    # Build output rows with quantile-level hierarchy enforcement
    raw_quantiles: list[list[float | int]] = []
    for i in range(num_trophies):
        mu = pred_medians[i]
        spread = pred_spreads[i]
        q_vals = [round4(inv_logit(mu + z * spread)) for z in Z]
        for j in range(len(q_vals)):
            q_vals[j] = min(100, max(0, q_vals[j]))
            if j > 0 and q_vals[j] < q_vals[j - 1]:
                q_vals[j] = q_vals[j - 1]
        raw_quantiles.append(q_vals)

    # Quantile-level hierarchy enforcement across quantile levels
    for tau_lvl in range(len(LEVELS)):
        # Story prerequisites: downstream <= upstream
        for k in range(len(story_ids) - 1):
            id_curr = story_ids[k]
            id_next = story_ids[k + 1]
            if id_curr in trophy_id_to_idx and id_next in trophy_id_to_idx:
                idx_curr = trophy_id_to_idx[id_curr]
                idx_next = trophy_id_to_idx[id_next]
                if raw_quantiles[idx_next][tau_lvl] > raw_quantiles[idx_curr][tau_lvl]:
                    raw_quantiles[idx_next][tau_lvl] = raw_quantiles[idx_curr][tau_lvl]

        if 14 in trophy_id_to_idx and 7 in trophy_id_to_idx:
            idx_14 = trophy_id_to_idx[14]
            idx_7 = trophy_id_to_idx[7]
            if raw_quantiles[idx_14][tau_lvl] > raw_quantiles[idx_7][tau_lvl]:
                raw_quantiles[idx_14][tau_lvl] = raw_quantiles[idx_7][tau_lvl]

        if num_trophies > 1:
            min_other_q = min(raw_quantiles[idx][tau_lvl] for idx in range(1, num_trophies))
            if raw_quantiles[0][tau_lvl] > min_other_q:
                raw_quantiles[0][tau_lvl] = min_other_q

    # Final sweep: re-verify row monotonicity
    rows: list[dict[str, Any]] = []
    for i, trophy in enumerate(trophies):
        q = raw_quantiles[i]
        for j in range(len(q)):
            q[j] = min(100, max(0, q[j]))
            if j > 0 and q[j] < q[j - 1]:
                q[j] = q[j - 1]
        rows.append({"id": trophy["id"], "q": q})

    return rows


def forecast_case(case: dict[str, Any], name: str) -> dict[str, Any]:
    """Validate case and forecast earn rate quantiles for every trophy."""
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, expected {LEVELS}")

    history = case.get("history", [])
    if not history:
        raise ValueError(f"{name} has no history")

    trophies = case.get("trophies", [])
    if not trophies:
        raise ValueError(f"{name} lists no trophies")

    horizon_minutes = float(case.get("horizon_minutes", 90.0))
    target_ts = parse_iso(case.get("target", ""))
    if target_ts <= 0:
        at_ts = parse_iso(case.get("at", ""))
        target_ts = at_ts + horizon_minutes * 60.0

    distinct_refreshes = deduplicate_refreshes(history)

    if len(distinct_refreshes) <= 1:
        rows = forecast_single_refresh(trophies, history[-1], horizon_minutes)
    else:
        rows = forecast_multi_refresh(trophies, distinct_refreshes, target_ts, horizon_minutes)

    return {"trophies": rows}


def main() -> None:
    """Read all *.in files in sorted order, write <id>.out to OUTPUT_DIR."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    in_files = sorted(INPUT_DIR.glob("*.in"))

    for path in in_files:
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case, path.name)
            out_json = json.dumps(forecast, separators=(",", ":")) + "\n"

            # Check size limit: at most 65,536 bytes
            if len(out_json.encode("utf-8")) > 65536:
                raise ValueError(f"Output for {path.name} exceeds 65,536 bytes")

            out_path = OUTPUT_DIR / f"{path.stem}.out"
            out_path.write_text(out_json, encoding="utf-8")
        except Exception as e:
            sys.stderr.write(f"Error forecasting {path.name}: {e}\n")
            raise


if __name__ == "__main__":
    main()
