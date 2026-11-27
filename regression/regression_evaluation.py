"""Evaluación común de los modelos de regresión disponibles.

Crea una tabla con los resultados de todos los modelos que ya se hayan ejecutado.
También calcula bootstrap pareado entre RNA+CNA y cada modalidad simple para
Polinómica, Huber y Theil-Sen, usando las predicciones guardadas del test.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SEED = 42
N_BOOTSTRAP = 20000

FILES = {
    "LinearRegression": "linear_regression_results.csv",
    "Ridge": "ridge_results.csv",
    "RANSAC": "ransac_results.csv",
    "Polynomial": "polynomial_regression_results.csv",
    "Huber": "huber_regression_results.csv",
    "TheilSen": "theilsen_regression_results.csv",
}


def main():
    frames = []
    needed = {"scenario", "test_rmse", "test_mae", "test_r2"}
    for label, filename in FILES.items():
        path = RESULTS / filename
        if not path.exists():
            print(f"Pendiente: {filename}")
            continue
        df = pd.read_csv(path)
        if not needed.issubset(df.columns):
            print(f"Columnas insuficientes en {filename}: {needed - set(df.columns)}")
            continue
        df["model"] = label
        frames.append(df)

    if not frames:
        print("No hay resultados guardados para comparar.")
        return

    comparison = pd.concat(frames, ignore_index=True)
    baseline_path = RESULTS / "regression_baseline.csv"
    if baseline_path.exists():
        best_baseline = pd.read_csv(baseline_path)["rmse"].min()
        comparison["beats_baseline_rmse"] = comparison["test_rmse"] < best_baseline
        print(f"Mejor baseline RMSE: {best_baseline:.4f}")

    cols = ["model", "scenario", "cv_rmse_mean", "cv_rmse_std",
            "test_rmse", "test_mae", "test_r2", "test_spearman",
            "beats_baseline_rmse"]
    comparison = comparison[[c for c in cols if c in comparison.columns]]
    comparison = comparison.sort_values("test_rmse")
    comparison.to_csv(RESULTS / "regression_comparison.csv", index=False)
    print("\n=== Resultados (menor RMSE es mejor) ===")
    print(comparison.round(4).to_string(index=False))

    boot_rows = []
    for name in ["polynomial", "huber", "theilsen"]:
        path = RESULTS / f"{name}_regression_predictions.csv"
        if not path.exists():
            continue
        preds = pd.read_csv(path)
        for other in ["RNA", "CNA"]:
            combined = preds[preds["scenario"] == "RNA+CNA"]
            simple = preds[preds["scenario"] == other]
            if combined.empty or simple.empty:
                continue
            a = combined[["sample_id", "y_true", "y_pred"]].merge(
                simple[["sample_id", "y_true", "y_pred"]],
                on="sample_id", suffixes=("_combined", "_simple"),
                validate="one_to_one")
            if len(a) != len(combined) or len(a) != len(simple):
                raise ValueError(f"Predicciones no alineadas: {name} {other}")
            if not np.allclose(a["y_true_combined"], a["y_true_simple"]):
                raise ValueError("Las etiquetas reales no coinciden")
            rng = np.random.default_rng(SEED)
            y = a["y_true_combined"].to_numpy()
            pc = a["y_pred_combined"].to_numpy()
            ps = a["y_pred_simple"].to_numpy()
            n = len(a)
            diffs_rmse = np.empty(N_BOOTSTRAP)
            diffs_mae = np.empty(N_BOOTSTRAP)
            for i in range(N_BOOTSTRAP):
                indices = rng.integers(0, n, size=n)
                e_c = y[indices] - pc[indices]
                e_s = y[indices] - ps[indices]
                diffs_rmse[i] = np.sqrt(np.mean(e_c**2)) - np.sqrt(np.mean(e_s**2))
                diffs_mae[i] = np.mean(np.abs(e_c)) - np.mean(np.abs(e_s))
            boot_rows.append({
                "model": name, "comparison": f"RNA+CNA vs {other}",
                "n_samples": n,
                "rmse_difference_mean": diffs_rmse.mean(),
                "rmse_ci_lower": np.percentile(diffs_rmse, 2.5),
                "rmse_ci_upper": np.percentile(diffs_rmse, 97.5),
                "mae_difference_mean": diffs_mae.mean(),
                "mae_ci_lower": np.percentile(diffs_mae, 2.5),
                "mae_ci_upper": np.percentile(diffs_mae, 97.5),
            })

    if boot_rows:
        bootstrap = pd.DataFrame(boot_rows)
        bootstrap.to_csv(RESULTS / "regression_bootstrap_34.csv", index=False)
        print("\n=== Bootstrap pareado (diferencia negativa favorece RNA+CNA) ===")
        print(bootstrap.round(4).to_string(index=False))
    else:
        print("\nSin predicciones de nuestros tres modelos para hacer bootstrap.")


if __name__ == "__main__":
    main()
