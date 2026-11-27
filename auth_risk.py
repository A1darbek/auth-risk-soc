"""Offline experiment for a synthetic authentication log."""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
FEATURES = ["failures_15m", "new_device", "new_region", "off_hours", "client", "mfa"]
NUMERIC = ["failures_15m"]
CATEGORICAL = FEATURES[1:]
TITLES = {
    "failures_15m": "Ошибки за 15 мин", "new_device": "Новое устройство",
    "new_region": "Новый регион", "off_hours": "Вне рабочего времени",
    "client": "Тип клиента", "mfa": "Результат MFA",
}
# Each row defines historical failure volume, context probabilities, and client/MFA mixes.
PROFILES = {
    "routine": (0.55, .13, .06, .15, (.58, .39, .03), (.93, .04, .03)),
    "travel": (1.3, .59, .65, .40, (.55, .43, .02), (.85, .08, .07)),
    "support": (4.4, .25, .10, .23, (.61, .28, .11), (.74, .19, .07)),
    "burst": (4.2, .57, .38, .42, (.42, .19, .39), (.47, .35, .18)),
    "unfamiliar": (1.15, .72, .58, .47, (.64, .27, .09), (.59, .22, .19)),
    "quiet": (.75, .29, .22, .29, (.57, .36, .07), (.79, .12, .09)),
}
NAMES = {"baseline": "Частый класс", "logistic": "Логистическая регрессия", "forest": "Случайный лес"}
SEED = 42


def generate():
    rng = np.random.default_rng(SEED)
    rows = []
    start = pd.Timestamp("2025-01-01", tz="UTC")
    for day in range(90):
        drift = day >= 72
        for minute in np.sort(rng.choice(1440, 240, replace=False)):
            label = int(rng.random() < (.033 if drift else .052))
            if label:
                profile = rng.choice(["burst", "unfamiliar", "quiet"],
                                     p=[.31, .39, .30] if drift else [.48, .37, .15])
            else:
                profile = rng.choice(["routine", "travel", "support"],
                                     p=[.78, .15, .07] if drift else [.85, .10, .05])
            volume, device, region, hours, client, mfa = PROFILES[profile]
            rows.append([
                f"E{len(rows):05d}", start + pd.Timedelta(days=day, minutes=int(minute)),
                int(rng.poisson(volume)), "yes" if rng.random() < device else "no",
                "yes" if rng.random() < region else "no",
                "yes" if rng.random() < hours else "no",
                rng.choice(["browser", "mobile", "service"], p=client),
                rng.choice(["passed", "denied", "unavailable"], p=mfa), profile, label,
            ])
    raw = pd.DataFrame(rows, columns=["event_id", "timestamp", *FEATURES, "scenario", "label"])
    for col in FEATURES:
        raw.loc[rng.random(len(raw)) < .025, col] = np.nan
    raw.loc[rng.choice(raw.index, 80, replace=False), "failures_15m"] = -1
    return pd.concat([raw, raw.sample(120, random_state=SEED)], ignore_index=True)


def clean(frame):
    result = frame.copy()
    counts = pd.to_numeric(result["failures_15m"], errors="coerce")
    result["failures_15m"] = counts.where(counts.ge(0) & np.isfinite(counts) & counts.mod(1).eq(0))
    for col in CATEGORICAL:
        values = result[col].astype("str").str.strip().str.lower().astype(object)
        result[col] = values.where(values.notna() & values.ne(""), np.nan)
    if "event_id" in result:
        result = result.drop_duplicates("event_id")
    if "timestamp" in result:
        result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
        result = result.sort_values("timestamp")
    return result


def pipeline(estimator):
    preprocess = ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC),
        ("categorical", make_pipeline(SimpleImputer(strategy="constant", fill_value="missing"),
                                      OneHotEncoder(handle_unknown="ignore")), CATEGORICAL),
    ])
    return make_pipeline(preprocess, estimator)


def metrics(y, scores, threshold):
    pred = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(accuracy=accuracy_score(y, pred), precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred, zero_division=0), f1=f1_score(y, pred, zero_division=0),
                ap=average_precision_score(y, scores), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                alerts=int(pred.sum()))


def save_chart(fig, output, name):
    fig.tight_layout()
    fig.savefig(output / f"{name}.png", dpi=150, facecolor="white")
    plt.close(fig)


def train(data_dir, output):
    data_dir.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    raw = generate()
    raw.to_csv(data_dir / "raw.csv", index=False)
    data = clean(raw)
    age = (data["timestamp"].dt.normalize() - data["timestamp"].min().normalize()).dt.days
    data["split"] = np.select([age < 54, age < 72], ["train", "validation"], default="test")
    data.to_csv(data_dir / "clean.csv", index=False)
    training, validation, test = (data[data.split.eq(part)] for part in ["train", "validation", "test"])
    models = {
        "baseline": pipeline(DummyClassifier(strategy="most_frequent")),
        "logistic": pipeline(LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)),
        "forest": pipeline(RandomForestClassifier(n_estimators=200, min_samples_leaf=25, max_depth=8,
                                                  class_weight="balanced", random_state=SEED, n_jobs=2)),
    }
    results = {}
    for name, model in models.items():
        model.fit(training[FEATURES], training.label)
        val_scores = model.predict_proba(validation[FEATURES])[:, 1]
        candidates = [.5] if name == "baseline" else [i / 100 for i in range(5, 96)]
        threshold = max(candidates, key=lambda t: (f1_score(validation.label, val_scores >= t,
                                                            zero_division=0), t))
        results[name] = {"threshold": float(threshold),
                         "validation": metrics(validation.label, val_scores, threshold)}
    winner = max(["logistic", "forest"], key=lambda name: results[name]["validation"]["f1"])
    test_scores = {}
    for name, model in models.items():
        test_scores[name] = model.predict_proba(test[FEATURES])[:, 1]
        results[name]["test"] = metrics(test.label, test_scores[name], results[name]["threshold"])

    model, threshold = models[winner], results[winner]["threshold"]
    x_test = test[FEATURES]
    rng = np.random.default_rng(117)
    jitter = x_test.copy()
    jitter["failures_15m"] = (jitter["failures_15m"] + rng.integers(-2, 3, len(test))).clip(lower=0)
    dropout = x_test.copy()
    for col in ["new_device", "new_region", "mfa"]:
        dropout.loc[rng.random(len(test)) < .35, col] = np.nan
    stress = {}
    for name, frame in {"Исходный тест": x_test, "Шум счётчика": jitter, "Потеря контекста": dropout}.items():
        scores = model.predict_proba(frame)[:, 1]
        stress[name] = metrics(test.label, scores, threshold)
        stress[name]["flips"] = int(((scores >= threshold) != (test_scores[winner] >= threshold)).sum())
    importance = permutation_importance(model, x_test, test.label, scoring="average_precision",
                                        n_repeats=8, random_state=SEED)
    order = np.argsort(importance.importances_mean)
    reference = {col: training[col].mode().iloc[0] for col in FEATURES}
    reference["failures_15m"] = float(training["failures_15m"].median())
    joblib.dump({"model": model, "name": winner, "threshold": threshold, "reference": reference},
                output / "model.joblib")
    predictions = test[["event_id", "timestamp", "scenario", "label"]].copy()
    predictions["score"] = test_scores[winner]
    predictions["alert"] = (predictions.score >= threshold).astype(int)
    predictions.to_csv(output / "test_predictions.csv", index=False)
    demo_rows = []
    for label, alert, name, rank_up in [(0, 0, "TN", True), (1, 1, "TP", False),
                                        (0, 1, "FP", True), (1, 0, "FN", False)]:
        matches = predictions[(predictions.label == label) & (predictions.alert == alert)]
        if not matches.empty:
            index = matches.sort_values("score", ascending=rank_up).index[0]
            demo_rows.append({**test.loc[index, FEATURES].to_dict(),
                              "event_id": test.at[index, "event_id"], "demo_case": name})
    pd.DataFrame(demo_rows, columns=["event_id", *FEATURES, "demo_case"]).to_csv(
        data_dir / "demo.csv", index=False)

    error_table = predictions.assign(fp=(predictions.label == 0) & (predictions.alert == 1),
                                     fn=(predictions.label == 1) & (predictions.alert == 0))
    errors = error_table.groupby("scenario").agg(n=("label", "size"), fp=("fp", "sum"), fn=("fn", "sum"))
    summary = {
        "raw": len(raw), "clean": len(data), "winner": winner,
        "splits": {key: {"n": len(frame), "positive": int(frame.label.sum()),
                         "start": str(frame.timestamp.min().date()), "end": str(frame.timestamp.max().date())}
                   for key, frame in zip(["train", "validation", "test"], [training, validation, test])},
        "missing": {col: int(data[col].isna().sum()) for col in FEATURES},
        "models": results, "stress": stress,
        "importance": {FEATURES[i]: {"mean": float(importance.importances_mean[i]),
                                     "std": float(importance.importances_std[i])} for i in order[::-1]},
        "errors": errors.to_dict(orient="index"),
    }
    (output / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for col in ["failures_15m", "mfa"]:
        pd.crosstab(training[col].fillna(-1) if col == "failures_15m" else training[col].fillna("missing"),
                    training.label).to_csv(output / f"eda_{col}.csv")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    parts = summary["splits"]
    axes[0].bar(["Обучение", "Валидация", "Тест"], [100 * p["positive"] / p["n"] for p in parts.values()],
                color=["#276e86", "#34889d", "#e08b45"])
    axes[0].set_ylabel("Подозрительные события, %")
    axes[0].set_title("Доля положительного класса")
    mfa_rates = pd.crosstab(training.mfa.fillna("missing"), training.label, normalize="columns")
    mfa_rates.columns = ["Легитимные", "Подозрительные"]
    mfa_rates.plot.bar(ax=axes[1], rot=15, color=["#276e86", "#e08b45"])
    axes[1].set_title("MFA в обучающей выборке")
    axes[1].set_ylabel("Доля внутри класса")
    axes[1].set_xlabel("")
    save_chart(fig, output, "eda")

    fig, ax = plt.subplots(figsize=(7, 3.6))
    for name, scores in test_scores.items():
        precision, recall, _ = precision_recall_curve(test.label, scores)
        ax.plot(recall, precision, label=f"{NAMES[name]} · AP={results[name]['test']['ap']:.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.03))
    ax.legend()
    save_chart(fig, output, "precision_recall")

    fig, ax = plt.subplots(figsize=(5.3, 3.5))
    ConfusionMatrixDisplay.from_predictions(test.label, test_scores[winner] >= threshold,
                                            display_labels=["Легитимный", "Подозрительный"],
                                            cmap="Blues", ax=ax, colorbar=False, values_format="d")
    ax.set(xlabel="Решение модели", ylabel="Истинная метка")
    save_chart(fig, output, "confusion")

    fig, ax = plt.subplots(figsize=(7, 3.1))
    ax.barh([TITLES[FEATURES[i]] for i in order], importance.importances_mean[order],
            xerr=importance.importances_std[order], color="#276e86", capsize=3)
    ax.set_xlabel("Падение Average Precision после перемешивания")
    save_chart(fig, output, "importance")
    return summary


def explain(bundle, row):
    model = bundle["model"]
    score = float(model.predict_proba(row)[0, 1])
    contrasts = {}
    for col, typical in bundle["reference"].items():
        changed = row.copy()
        changed[col] = typical
        contrasts[col] = round(score - float(model.predict_proba(changed)[0, 1]), 4)
    return {"score": round(score, 6), "threshold": bundle["threshold"],
            "alert": bool(score >= bundle["threshold"]), "missing": row.columns[row.isna().iloc[0]].tolist(),
            "contrast": dict(sorted(contrasts.items(), key=lambda pair: -abs(pair[1])))}


def detect(path, model_path):
    bundle = joblib.load(model_path)
    events = pd.read_csv(path)
    missing_columns = set(FEATURES) - set(events.columns)
    if missing_columns:
        raise ValueError(f"Отсутствуют столбцы {sorted(missing_columns)}")
    for index, row in clean(events).iterrows():
        result = explain(bundle, row[FEATURES].to_frame().T)
        print(json.dumps({"event_id": str(row.get("event_id", index)), **result}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="Оценка риска по событиям аутентификации")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("train")
    run.add_argument("--data-dir", type=Path, default=ROOT / "data")
    run.add_argument("--output", type=Path, default=ROOT / "artifacts")
    infer = sub.add_parser("detect")
    infer.add_argument("csv", type=Path)
    infer.add_argument("--model", type=Path, default=ROOT / "artifacts/model.joblib")
    args = parser.parse_args()
    if args.command == "train":
        summary = train(args.data_dir, args.output)
        print(json.dumps(summary["models"], ensure_ascii=False, indent=2))
    else:
        detect(args.csv, args.model)


if __name__ == "__main__":
    main()
