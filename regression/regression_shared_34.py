"""Funciones comunes para Polinómica, Huber y Theil-Sen.

Usa el preprocesamiento y la división de pacientes existentes del proyecto.
Ejecutar los scripts desde cualquier carpeta; las rutas parten de la raíz del repositorio.
"""
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import HuberRegressor, LinearRegression, TheilSenRegressor
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "Preprocessing"))
from filters import ModalFrequencyFilter, ZeroFractionFilter
from regression_data import load_regression_data, regression_metrics

SEED = 42
N_TRIALS = 30
N_FOLDS = 5
CNA_KIND = "disc"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_pipeline(scenario, X, name):
    # Los filtros, el escalado y la selección se ajustan en CADA fold.
    if scenario == "RNA":
        initial = ("filter", ZeroFractionFilter(threshold=0.95))
    elif scenario == "CNA":
        initial = ("filter", ModalFrequencyFilter(threshold=0.95))
    else:
        initial = ("filter", ColumnTransformer([
            ("rna", ZeroFractionFilter(threshold=0.95),
             [c for c in X.columns if c.startswith("RNA_")]),
            ("cna", ModalFrequencyFilter(threshold=0.95),
             [c for c in X.columns if c.startswith("CNA_")]),
        ]))

    steps = [initial,
             ("scale", StandardScaler()),
             ("reduce", SelectKBest(score_func=f_regression, k=5))]

    if name == "polynomial":
        steps.extend([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("model", LinearRegression()),
        ])
    elif name == "huber":
        steps.append(("model", HuberRegressor(max_iter=2000)))
    elif name == "theilsen":
        # Pocas variables: Theil-Sen es muy costoso en problemas con miles de genes.
        steps.append(("model", TheilSenRegressor(
            random_state=SEED, max_subpopulation=100, n_jobs=1)))
    else:
        raise ValueError(f"Modelo desconocido: {name}")
    return Pipeline(steps)


def trial_parameters(trial, name):
    if name == "polynomial":
        return {
            "reduce__k": trial.suggest_int("k", 3, 10),
            "poly__degree": trial.suggest_int("degree", 2, 3),
        }
    if name == "huber":
        return {
            "reduce__k": trial.suggest_int("k", 5, 40),
            "model__epsilon": trial.suggest_float("epsilon", 1.1, 2.5),
            "model__alpha": trial.suggest_float("alpha", 1e-5, 1.0, log=True),
        }
    if name == "theilsen":
        return {
            "reduce__k": trial.suggest_int("k", 3, 8),
            "model__max_subpopulation": trial.suggest_categorical(
                "max_subpopulation", [50, 100, 200]),
        }
    raise ValueError(name)


def train_model(name):
    RESULTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)

    data = load_regression_data(cna_kind=CNA_KIND)
    y = data["y"]
    is_train = data["is_train"]
    groups = data["patient_ids"].to_numpy()
    assert set(groups[is_train]).isdisjoint(set(groups[~is_train])), \
        "Hay pacientes repetidos entre train y test"

    y_train, y_test = y[is_train], y[~is_train]
    groups_train = groups[is_train]
    cv = GroupKFold(n_splits=N_FOLDS)

    rows, fold_rows, prediction_rows = [], [], []

    for scenario, X in data["datasets"].items():
        print(f"\n=== {name.upper()} - {scenario} ===", flush=True)
        X_train, X_test = X.iloc[is_train], X.iloc[~is_train]
        pipe = make_pipeline(scenario, X, name)
        started = time.time()

        def objective(trial):
            params = trial_parameters(trial, name)
            candidate = clone(pipe).set_params(**params)
            scores = cross_validate(
                candidate, X_train, y_train, cv=cv, groups=groups_train,
                scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"},
                error_score="raise",
            )
            rmse = (-scores["test_rmse"]).tolist()
            trial.set_user_attr("fold_rmse", rmse)
            trial.set_user_attr("fold_r2", scores["test_r2"].tolist())
            return float(np.mean(rmse))

        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))

        def report(study, trial):
            print(f"  prueba {trial.number + 1}/{N_TRIALS}: "
                  f"RMSE={trial.value:.4f} | {time.time()-started:.0f}s", flush=True)

        study.optimize(objective, n_trials=N_TRIALS, callbacks=[report])
        best_params = trial_parameters(
            optuna.trial.FixedTrial(study.best_params), name)
        fitted = clone(pipe).set_params(**best_params).fit(X_train, y_train)

        pred_train = fitted.predict(X_train)
        pred_test = fitted.predict(X_test)
        train_m = regression_metrics(y_train, pred_train)
        test_m = regression_metrics(y_test, pred_test)
        fold_rmse = study.best_trial.user_attrs["fold_rmse"]
        fold_r2 = study.best_trial.user_attrs["fold_r2"]
        safe = scenario.replace("+", "_")

        joblib.dump(fitted, MODELS / f"{name}_regression_{safe}.joblib")
        study.trials_dataframe().to_csv(
            RESULTS / f"{name}_regression_trials_{safe}.csv", index=False)

        for i, (rmse, r2) in enumerate(zip(fold_rmse, fold_r2), start=1):
            fold_rows.append({"scenario": scenario, "model": name,
                              "fold": i, "rmse": rmse, "r2": r2})

        prediction_rows.append(pd.DataFrame({
            "scenario": scenario, "sample_id": X_test.index,
            "patient_id": groups[~is_train], "y_true": y_test.to_numpy(),
            "y_pred": pred_test,
        }))
        rows.append({
            "scenario": scenario, "model": name, "n_trials": N_TRIALS,
            "cna_kind": CNA_KIND, "best_params": str(study.best_params),
            "cv_rmse_mean": float(np.mean(fold_rmse)),
            "cv_rmse_std": float(np.std(fold_rmse)),
            "cv_r2_mean": float(np.mean(fold_r2)),
            "train_rmse": train_m["rmse"], "train_r2": train_m["r2"],
            "test_rmse": test_m["rmse"], "test_mae": test_m["mae"],
            "test_r2": test_m["r2"], "test_spearman": test_m["spearman"],
        })
        print("Mejores parámetros:", study.best_params)
        print(f"CV RMSE: {np.mean(fold_rmse):.4f} ± {np.std(fold_rmse):.4f}")
        print(f"TEST RMSE: {test_m['rmse']:.4f} | MAE: {test_m['mae']:.4f} "
              f"| R2: {test_m['r2']:.4f} | Spearman: {test_m['spearman']:.4f}")

        # Guardar al terminar cada escenario para no perder resultados.
        pd.DataFrame(rows).to_csv(RESULTS / f"{name}_regression_results.csv", index=False)
        pd.DataFrame(fold_rows).to_csv(RESULTS / f"{name}_regression_cv_folds.csv", index=False)
        pd.concat(prediction_rows, ignore_index=True).to_csv(
            RESULTS / f"{name}_regression_predictions.csv", index=False)

    baseline_path = RESULTS / "regression_baseline.csv"
    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path)
        best_rmse = baseline["rmse"].min()
        print(f"\nBaseline RMSE: {best_rmse:.4f}")
        for row in rows:
            print(f"{row['scenario']}: mejora baseline = {row['test_rmse'] < best_rmse}")
    else:
        print("\nNo se encuentra regression_baseline.csv: ejecuta regression_baseline.py")
