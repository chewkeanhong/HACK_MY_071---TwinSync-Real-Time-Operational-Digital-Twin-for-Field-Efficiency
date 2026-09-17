"""Train the 7-day failure risk model.

    python scripts/train_risk_model.py --out models/

=============================================================================
READ THIS FIRST: the training data is synthetic, and here is exactly how.
=============================================================================

There is no public dataset of telecom site failures. Operators do not publish their
outage histories, and a hackathon cannot obtain one. So the labels here are generated
from an explicit hazard model, written out below in full so it can be argued with rather
than taken on trust.

What that does and does not buy:

* The model, the training procedure, the leakage-controlled validation and the reported
  metrics are all real. If the hazard model is right, the risk scores are useful.
* The metrics measure how well LightGBM recovers *this hazard function* from noisy
  samples. They are NOT evidence about real-world failures, and the AUC below should
  never be quoted as if they were.
* Every term is drawn from published reliability physics rather than invented, and each
  one carries its source in a comment. That is the most this can honestly claim.

THE HAZARD MODEL
----------------

Daily failure hazard is a product of independent multiplicative stressors on a small
baseline rate, which is the standard proportional-hazards shape. P(failure within seven
days) then follows from the exponential survival function.

1.  **Thermal ageing -- Arrhenius.** Semiconductor and electrolytic-capacitor wear-out
    rate roughly doubles per 10 degC of operating temperature. This is the single
    best-established term here and the reason `peak_temperature_c` dominates.

2.  **Thermal cycling -- Coffin-Manson.** Solder-joint fatigue scales with the number of
    thermal cycles to a fractional power, not linearly; damage accumulates sub-linearly.

3.  **Age -- increasing-hazard Weibull.** Shape > 1, i.e. old kit fails more. Infant
    mortality is deliberately not modelled: this is a mature CBD fleet.

4.  **Deferred maintenance.** Hazard climbs with time since last service, saturating --
    a site two years overdue is not twice as bad as one a year overdue.

5.  **VSWR trend.** A rising standing-wave ratio means a degrading feeder, connector or
    antenna. In practice this is the strongest *leading* indicator an operator has, and
    it is weighted accordingly.

6.  **Lightning.** Strike attachment probability rises with structure height and with
    terrain elevation, so this term multiplies strike density by both. Peninsular
    Malaysia has among the highest ground-flash densities in the world, which is why
    this matters here and would not in, say, Helsinki.

7.  **Moisture ingress.** Rain and humidity drive corrosion, and corrosion interacts
    with age -- new gaskets keep water out, ten-year-old ones do not. Modelled as a
    product rather than a sum for that reason.

8.  **Terrain slope.** A proxy for exposure: steeper ground means more wind loading at
    the mast, worse drainage at the base, and harder access for preventive work.

9.  **Load factor.** Sustained high utilisation means sustained heat, on top of ambient.

10. **Vegetation encroachment.** Physical contact risk with the feeder run and the
    access track.

Sites are generated with correlated attributes (a site that is old also tends to be
overdue for maintenance), because independent features would make the learning problem
artificially easy and inflate every metric reported below.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from twinsync.risk import FEATURE_NAMES  # noqa: E402

N_SITES = 2500
WEEKS_PER_SITE = 20
HORIZON_DAYS = 7.0

# Baseline daily hazard for a new, cool, well-maintained site. About one failure per
# site per 14 years before any stressor is applied.
BASE_DAILY_HAZARD = 2.0e-4

ARRHENIUS_REFERENCE_C = 40.0
ARRHENIUS_DOUBLING_C = 10.0


def generate(n_sites: int, weeks: int,
             seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthesise records from the hazard model above.

    Returns ``(features, labels, site_ids, true_probability)``. The true probability is
    kept because it gives the Bayes-optimal ceiling for this labelling process -- see
    the note where it is used. Nothing downstream of training is allowed to see it.
    """
    rng = np.random.default_rng(seed)
    rows, labels, groups, truth = [], [], [], []

    for site in range(n_sites):
        # -- site attributes, deliberately correlated -------------------
        age = float(np.clip(rng.gamma(4.0, 2.2), 0.5, 25.0))
        # Older sites tend to be further behind on maintenance.
        maintenance = float(np.clip(
            rng.normal(120.0 + 12.0 * age, 70.0), 5.0, 900.0))
        load = float(np.clip(rng.beta(5.0, 3.0), 0.1, 1.0))
        elevation = float(np.clip(rng.normal(45.0, 14.0), 5.0, 120.0))
        slope = float(np.clip(rng.gamma(1.6, 2.2), 0.0, 35.0))
        antenna_height = float(np.clip(rng.gamma(3.0, 22.0), 12.0, 300.0))
        subscribers = float(np.clip(rng.gamma(2.2, 3600.0), 200.0, 60000.0))
        encroachment = float(np.clip(rng.beta(2.0, 3.0), 0.0, 1.0))
        # A degrading feeder drifts up with age, with wide spread.
        vswr_trend = float(np.clip(
            rng.gamma(1.5, 0.10) + 0.012 * age, 0.0, 1.6))

        for _ in range(weeks):
            # -- weekly conditions ---------------------------------------
            # Monsoon weeks are wet; most weeks are not. A bimodal mixture rather than
            # a normal, because tropical rainfall genuinely is bimodal.
            monsoon = rng.random() < 0.35
            rainfall = float(np.clip(
                rng.gamma(2.0, 14.0) if monsoon else rng.gamma(1.2, 2.5), 0.0, 180.0))
            lightning = float(np.clip(
                rng.gamma(2.0, 1.5) if monsoon else rng.gamma(1.0, 0.35), 0.0, 24.0))
            humidity = float(np.clip(
                rng.normal(88.0 if monsoon else 76.0, 5.0), 45.0, 100.0))

            ambient = 32.0 + 4.0 * rng.normal()
            # Utilisation heat plus solar gain, minus evaporative relief when wet.
            peak_temp = float(np.clip(
                ambient + 18.0 * load + 6.0 * rng.random() - 0.02 * rainfall,
                18.0, 95.0))
            cycles = float(np.clip(rng.normal(70.0 + 40.0 * load, 25.0), 0.0, 400.0))

            # -- hazard terms --------------------------------------------
            # 1. Arrhenius: doubling per ARRHENIUS_DOUBLING_C above reference.
            thermal = 2.0 ** ((peak_temp - ARRHENIUS_REFERENCE_C)
                              / ARRHENIUS_DOUBLING_C)
            # 2. Coffin-Manson: sub-linear accumulation of cycling damage.
            cycling = 1.0 + 0.6 * (cycles / 100.0) ** 0.6
            # 3. Weibull, shape 1.9.
            wear = 1.0 + (age / 12.0) ** 1.9
            # 4. Deferred maintenance, saturating.
            deferred = 1.0 + 0.9 * (maintenance / (maintenance + 240.0))
            # 5. VSWR trend -- the strongest leading indicator.
            feeder = 1.0 + 3.4 * vswr_trend
            # 6. Lightning attachment scales with height and elevation.
            strike = 1.0 + 0.05 * lightning * (antenna_height / 100.0) * (
                1.0 + elevation / 200.0)
            # 7. Moisture ingress, gated by seal age.
            ingress = 1.0 + 0.010 * rainfall * (humidity / 100.0) * (age / 12.0)
            # 8. Terrain exposure.
            exposure = 1.0 + 0.020 * slope
            # 9/10. Vegetation contact.
            vegetation = 1.0 + 0.35 * encroachment

            daily = (BASE_DAILY_HAZARD * thermal * cycling * wear * deferred
                     * feeder * strike * ingress * exposure * vegetation)
            probability = 1.0 - np.exp(-daily * HORIZON_DAYS)
            failed = int(rng.random() < probability)

            rows.append([
                age, maintenance, cycles, peak_temp, vswr_trend, load, subscribers,
                slope, elevation, antenna_height, rainfall, lightning, humidity,
                encroachment,
            ])
            labels.append(failed)
            groups.append(site)
            truth.append(probability)

    return (np.array(rows, dtype=np.float64),
            np.array(labels, dtype=np.int32),
            np.array(groups, dtype=np.int32),
            np.array(truth, dtype=np.float64))


def beeswarm(shap_values: np.ndarray, features: np.ndarray, names: list[str],
             path: Path, top: int = 12) -> None:
    """SHAP summary plot, drawn from LightGBM's own exact TreeSHAP output.

    Deliberately not using the `shap` package: `Booster.predict(pred_contrib=True)`
    already returns exact TreeSHAP values, so pulling in a large extra dependency to
    recompute the same numbers would only add install friction.
    """
    import matplotlib                                        # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt                          # noqa: PLC0415

    order = np.argsort(np.abs(shap_values).mean(axis=0))[::-1][:top][::-1]
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(9, 0.45 * len(order) + 1.8))
    for row, index in enumerate(order):
        values = shap_values[:, index]
        raw = features[:, index]
        # Colour by feature value, normalised to its own 5-95 percentile range so one
        # outlier cannot wash the whole row out.
        low, high = np.percentile(raw, [5, 95])
        shade = np.clip((raw - low) / max(high - low, 1e-9), 0.0, 1.0)
        jitter = rng.normal(0.0, 0.055, size=len(values))
        ax.scatter(values, row + jitter, c=shade, cmap="coolwarm", s=5,
                   alpha=0.5, linewidths=0)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[i] for i in order], fontsize=9)
    ax.axvline(0.0, color="#555", linewidth=0.8)
    ax.set_xlabel("SHAP value (log-odds contribution to 7-day failure risk)", fontsize=9)
    ax.set_title("Per-prediction feature attribution (exact TreeSHAP)\n"
                 "red = high feature value, blue = low", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.colorbar(plt.cm.ScalarMappable(cmap="coolwarm"), ax=ax,
                 label="feature value", pad=0.01).set_ticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def importance_plot(booster, path: Path) -> None:
    import matplotlib                                        # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt                          # noqa: PLC0415

    gains = booster.feature_importance(importance_type="gain")
    order = np.argsort(gains)
    names = [FEATURE_NAMES[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, 0.4 * len(names) + 1.5))
    ax.barh(range(len(names)), gains[order], color="#3f7fb5")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("total gain", fontsize=9)
    ax.set_title("LightGBM feature importance (gain)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train the 7-day risk model.")
    parser.add_argument("--out", default="models", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sites", type=int, default=N_SITES)
    parser.add_argument("--weeks", type=int, default=WEEKS_PER_SITE)
    args = parser.parse_args(argv)

    import lightgbm as lgb                                   # noqa: PLC0415
    from sklearn.metrics import (                            # noqa: PLC0415
        average_precision_score,
        brier_score_loss,
        roc_auc_score,
    )

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"generating {args.sites} sites x {args.weeks} weeks...")
    X, y, groups, p_true = generate(args.sites, args.weeks, args.seed)
    print(f"  {X.shape[0]:,} records, {y.sum():,} failures "
          f"({100 * y.mean():.2f}% positive)")

    # Split by SITE, not by row. Weeks from one site share its age, feeder condition and
    # terrain, so a random row split would put near-duplicates on both sides and report
    # an AUC that says more about the split than the model. Same discipline as
    # HeightImputer.cross_validate in scripts/impute_heights.py.
    rng = np.random.default_rng(args.seed)
    sites = np.unique(groups)
    rng.shuffle(sites)
    cut = int(0.75 * len(sites))
    train_sites = set(sites[:cut].tolist())
    is_train = np.array([g in train_sites for g in groups])
    print(f"  grouped split: {is_train.sum():,} train / {(~is_train).sum():,} test "
          f"across {len(sites)} sites")

    booster = lgb.train(
        {
            "objective": "binary",
            "metric": ["auc", "binary_logloss"],
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 120,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
            "verbosity": -1,
            "seed": args.seed,
        },
        lgb.Dataset(X[is_train], label=y[is_train], feature_name=FEATURE_NAMES),
        num_boost_round=600,
        valid_sets=[lgb.Dataset(X[~is_train], label=y[~is_train],
                                feature_name=FEATURE_NAMES)],
        callbacks=[lgb.early_stopping(40, verbose=False),
                   lgb.log_evaluation(100)],
    )
    print(f"  stopped at iteration {booster.best_iteration}")

    predicted = booster.predict(X[~is_train], num_iteration=booster.best_iteration)
    truth = y[~is_train]
    metrics = {
        "roc_auc": round(float(roc_auc_score(truth, predicted)), 4),
        "pr_auc": round(float(average_precision_score(truth, predicted)), 4),
        "brier": round(float(brier_score_loss(truth, predicted)), 6),
        "base_rate": round(float(truth.mean()), 5),
        "test_rows": int(truth.size),
        "test_positives": int(truth.sum()),
    }
    # The Bayes ceiling. Failure is a Bernoulli draw at ~2% per week, so even a model
    # that recovered the hazard function exactly could not score much above 0.70 -- the
    # remaining variance is the coin, not the model. Reporting the raw AUC without this
    # invites the reasonable objection "0.67 is barely better than guessing", when the
    # honest reading is that it captures most of what is knowable. Computed from the
    # generator's own probabilities, which training never sees.
    ceiling_auc = float(roc_auc_score(truth, p_true[~is_train]))
    ceiling_pr = float(average_precision_score(truth, p_true[~is_train]))
    captured = ((metrics["roc_auc"] - 0.5) / max(ceiling_auc - 0.5, 1e-9))
    metrics["bayes_ceiling_roc_auc"] = round(ceiling_auc, 4)
    metrics["bayes_ceiling_pr_auc"] = round(ceiling_pr, 4)
    metrics["fraction_of_achievable_lift"] = round(captured, 4)

    print(f"\n  held-out ROC-AUC {metrics['roc_auc']:.4f} | "
          f"PR-AUC {metrics['pr_auc']:.4f} (base rate {metrics['base_rate']:.4f}) | "
          f"Brier {metrics['brier']:.6f}")
    print(f"  Bayes ceiling for this labelling: ROC-AUC {ceiling_auc:.4f}, "
          f"PR-AUC {ceiling_pr:.4f}")
    print(f"  -> the model captures {100 * captured:.0f}% of the achievable lift "
          "above chance")

    # Calibration: does a predicted 3% actually fail 3% of the time? A ranking model
    # that is miscalibrated is still useless for "should we send someone this week".
    deciles = []
    edges = np.quantile(predicted, np.linspace(0, 1, 11))
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (predicted >= lo) & (predicted <= hi)
        if mask.sum() > 0:
            deciles.append({
                "predicted": round(float(predicted[mask].mean()), 5),
                "observed": round(float(truth[mask].mean()), 5),
                "n": int(mask.sum()),
            })
    metrics["calibration_deciles"] = deciles
    worst = max(abs(d["predicted"] - d["observed"]) for d in deciles)
    print(f"  worst calibration gap across deciles: {worst:.4f}")

    # -- artifacts -------------------------------------------------------

    model_path = args.out / "risk_lgbm.txt"
    booster.save_model(str(model_path), num_iteration=booster.best_iteration)
    print(f"\n  {model_path.name}: {model_path.stat().st_size / 1024:.0f} KB")

    importance_plot(booster, args.out / "risk_feature_importance.png")
    print("  risk_feature_importance.png")

    probe = X[~is_train][:4000]
    contributions = booster.predict(probe, pred_contrib=True,
                                    num_iteration=booster.best_iteration)
    beeswarm(contributions[:, :-1], probe, FEATURE_NAMES,
             args.out / "risk_shap_summary.png")
    print("  risk_shap_summary.png")

    (args.out / "risk_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    print("  risk_metrics.json")

    ranked = sorted(
        zip(FEATURE_NAMES, np.abs(contributions[:, :-1]).mean(axis=0)),
        key=lambda pair: -pair[1],
    )
    # Fleet-relative band cutoffs. "high" is the top decile of predicted risk, "medium"
    # the top third. A fixed cutoff on an absolute probability would colour every site
    # "low" for ever, because a few percent per week is what a healthy fleet looks like.
    bands = {
        "high": round(float(np.quantile(predicted, 0.90)), 5),
        "medium": round(float(np.quantile(predicted, 0.67)), 5),
        "basis": "quantiles of held-out predicted probability (p90 / p67)",
    }
    print(f"  bands: high >= {100 * bands['high']:.1f}%, "
          f"medium >= {100 * bands['medium']:.1f}% chance of failure in 7 days")

    meta = {
        "model": "lightgbm-v1",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature_names": FEATURE_NAMES,
        "horizon_days": HORIZON_DAYS,
        "bands": bands,
        "training_data": "synthetic, generated by scripts/train_risk_model.py",
        "n_records": int(X.shape[0]),
        "n_sites": int(args.sites),
        "best_iteration": int(booster.best_iteration),
        "metrics": {k: v for k, v in metrics.items() if k != "calibration_deciles"},
        "mean_abs_shap": {name: round(float(value), 5) for name, value in ranked},
        "sha256": sha256(model_path.read_bytes()).hexdigest(),
    }
    (args.out / "risk_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print("  risk_meta.json")

    print("\n  strongest drivers by mean |SHAP|:")
    for name, value in ranked[:5]:
        print(f"    {name:<28}{value:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
