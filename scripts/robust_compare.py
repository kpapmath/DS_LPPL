"""Pre-registered comparison of the robust variant with its baseline (docs/validation.md §8).

P1 steadiness: flicker (mean |Δ conf| between consecutive analysis dates) and number of alarm
   episodes (conf >= alarm after > 45 days off), baseline vs variant.
P2 skill: paired month-block bootstrap of the AP difference (variant - baseline) per label, on the
   same analysis dates. Primary label: major_top_126 (one-sided p).
Seed diagnostics: how many of the baseline's qualified windows survive the seed vote.

Usage: python scripts/robust_compare.py --base full-tuned --variant full-robust
Writes reports/robust_variant_analysis.{md,json}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lppl.config import PROJECT_ROOT
from lppl.data import db
from lppl.labels import compute_labels
from lppl.pipeline import resolve_run
from lppl.tuning.detection import average_precision
from lppl.validation import build_events, load_validation_config

PRIMARY = "major_top_126"


def episodes(ev: pd.DataFrame, alarm: float, gap_days: int = 45) -> int:
    hot = ev[ev["conf"] >= alarm].sort_values(["ticker", "date"])
    return int(sum((g["date"].diff().dt.days > gap_days).sum() + 1 for _, g in hot.groupby("ticker")))


def flicker(ev: pd.DataFrame) -> float:
    d = ev.sort_values(["ticker", "date"]).groupby("ticker")["conf"].diff().abs()
    return float(d.mean())


def paired_bootstrap(y, a, b, months, reps: int, seed: int) -> dict:
    """AP(b) - AP(a) with month blocks resampled jointly for both scores."""
    y, a, b = map(np.asarray, (y, a, b))
    codes, uniq = pd.factorize(months)
    rows_of = [np.flatnonzero(codes == k) for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    obs = average_precision(y, b) - average_precision(y, a)
    diffs = np.empty(reps)
    for r in range(reps):
        idx = np.concatenate([rows_of[k] for k in rng.integers(0, len(uniq), len(uniq))])
        diffs[r] = average_precision(y[idx], b[idx]) - average_precision(y[idx], a[idx])
    return {"ap_base": average_precision(y, a), "ap_variant": average_precision(y, b), "ap_diff": obs,
            "ci_low": float(np.nanquantile(diffs, 0.025)), "ci_high": float(np.nanquantile(diffs, 0.975)),
            "p_one_sided": float((np.sum(diffs <= 0) + 1) / (np.sum(~np.isnan(diffs)) + 1))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="full-tuned")
    ap.add_argument("--variant", default="full-robust")
    ap.add_argument("--validation-config", default=str(PROJECT_ROOT / "config" / "full_validation.yaml"))
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "reports" / "robust_variant_analysis"))
    args = ap.parse_args()

    s = load_validation_config(args.validation_config)
    eng = db.get_engine()
    runs = {k: resolve_run(eng, label=lab) for k, lab in (("base", args.base), ("variant", args.variant))}
    prices = db.load_all_prices(eng, "adj_close")
    ev = {k: build_events(db.load_indicator(eng, r), prices, s.events, s.split_date, s.embargo_days)
          for k, r in runs.items()}
    m = ev["base"][["ticker", "date", "conf"]].merge(ev["variant"][["ticker", "date", "conf"]],
                                                     on=["ticker", "date"], suffixes=("_base", "_variant"))
    out: dict = {"runs": runs, "n_dates_paired": len(m),
                 "n_dates": {k: len(v) for k, v in ev.items()}}

    # P1 steadiness
    out["P1"] = {k: {"flicker": flicker(v), "alarm_episodes": episodes(v, s.alarm_threshold),
                     "alarm_rate": float((v["conf"] >= s.alarm_threshold).mean()),
                     "mean_conf": float(v["conf"].mean())} for k, v in ev.items()}
    p1 = out["P1"]
    out["P1"]["passes"] = bool(p1["variant"]["flicker"] < p1["base"]["flicker"]
                               and p1["variant"]["alarm_episodes"] < p1["base"]["alarm_episodes"])

    # P2 skill per label, paired on the same dates
    ls = s.labels
    res = {}
    for name in ls.names:
        y = np.full(len(m), np.nan)
        for ticker, idx in m.groupby("ticker").groups.items():
            y[m.index.get_indexer(idx)] = compute_labels(prices[ticker], m.loc[idx, "date"], name,
                                                         ls.k_sigma, s.events.vol_window, ls.top_lookback)
        ok = ~np.isnan(y)
        d = m[ok]
        res[name] = {"n": int(ok.sum()), "events": int(np.nansum(y)),
                     **paired_bootstrap(y[ok], d["conf_base"], d["conf_variant"],
                                        pd.DatetimeIndex(d["date"]).to_period("M"), args.reps, s.seed)}
    out["P2"] = res
    out["P2"]["passes"] = bool(res[PRIMARY]["ap_diff"] > 0 and res[PRIMARY]["p_one_sided"] < 0.05)

    # seed diagnostics: baseline fits = seed 0 of the variant (same RNG)
    fb = db.load_fits(eng, runs["base"])[["ticker", "t2", "window_len", "qualified", "sign"]]
    fv = db.load_fits(eng, runs["variant"])[["ticker", "t2", "window_len", "qualified", "sign", "n_seeds_ok"]]
    j = fb.merge(fv, on=["ticker", "t2", "window_len"], suffixes=("_base", "_variant"))
    qb = j["qualified_base"] & (j["sign_base"] > 0)
    qv = j["qualified_variant"] & (j["sign_variant"] > 0)
    out["seeds"] = {"windows": len(j), "base_pos_qualified": int(qb.sum()), "variant_pos_qualified": int(qv.sum()),
                    "base_qualified_kept": float((qb & qv).sum() / max(qb.sum(), 1)),
                    "variant_only": int((qv & ~qb).sum()),
                    "n_seeds_ok_dist": {int(k): int(v) for k, v in
                                        fv["n_seeds_ok"].value_counts().sort_index().items()}}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out + ".json").write_text(json.dumps(out, indent=2, default=str))
    lines = [f"# Robust variant vs baseline\n\nbaseline `{runs['base']}` ({args.base}), "
             f"variant `{runs['variant']}` ({args.variant}), {len(m)} paired analysis dates\n",
             "## P1 steadiness\n\n| | flicker | alarm episodes | alarm rate | mean conf |\n|---|---|---|---|---|"]
    for k in ("base", "variant"):
        r = p1[k]
        lines.append(f"| {k} | {r['flicker']:.4f} | {r['alarm_episodes']} | {r['alarm_rate']:.3f} "
                     f"| {r['mean_conf']:.4f} |")
    lines += [f"\nP1 passes: **{p1['passes']}**\n", "## P2 AP per label (paired month-block bootstrap)\n",
              "| label | events | AP base | AP variant | diff | 95 % CI | p (one-sided) |",
              "|---|---|---|---|---|---|---|"]
    for name in ls.names:
        r = res[name]
        lines.append(f"| {name}{' (primary)' if name == PRIMARY else ''} | {r['events']} | {r['ap_base']:.4f} | "
                     f"{r['ap_variant']:.4f} | {r['ap_diff']:+.4f} | [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | "
                     f"{r['p_one_sided']:.3f} |")
    sd = out["seeds"]
    lines += [f"\nP2 passes: **{out['P2']['passes']}**\n", "## Seed vote\n",
              f"* positive qualified windows: baseline {sd['base_pos_qualified']}, "
              f"variant {sd['variant_pos_qualified']}",
              f"* share of the baseline's qualified windows kept by the vote: {sd['base_qualified_kept']:.1%}",
              f"* windows qualified only in the variant: {sd['variant_only']}",
              f"* agreeing-group size distribution: {sd['n_seeds_ok_dist']}"]
    Path(args.out + ".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
