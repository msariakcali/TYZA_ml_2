"""IBM Telco müşteri kaybı için uçtan uca sınıflandırma projesi.

Amaç:
    Telekom müşterilerinin hizmetten ayrılıp ayrılmayacağını (Churn) tahmin
    etmek; veri ön işleme, öznitelik mühendisliği, model karşılaştırma,
    çapraz doğrulama, hiperparametre ayarlama ve açıklanabilirlik adımlarını
    tekrarlanabilir tek bir akışta uygulamak.

Kullanılan başlıca kütüphaneler:
    pandas, numpy, scikit-learn, matplotlib, seaborn ve joblib.

Çalıştırma:
    1. Proje kökünde ``pip install -r requirements.txt`` komutunu çalıştırın.
    2. Ardından ``python src/train.py`` komutunu çalıştırın.
    3. Metrik, grafik ve model dosyaları ``outputs/`` klasörüne yazılır.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


RANDOM_STATE = 42
TARGET = "Churn"


def load_dataset(data_path: str | Path) -> pd.DataFrame:
    """CSV veri setini okur ve ham DataFrame'i döndürür."""
    return pd.read_csv(data_path)


def engineer_features(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Tip düzeltme ve anlamlı öznitelik üretimini gerçekleştirir.

    Eksik değerler burada doldurulmaz. Veri sızıntısını önlemek için sayısal
    medyan ve kategorik mod doldurma işlemleri model Pipeline'ı içinde,
    yalnızca eğitim verisine fit edilerek yapılır.
    """
    df = raw_df.copy(deep=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    missing_summary = (
        df.isna().sum()
        .rename("eksik_deger_sayisi")
        .to_frame()
        .query("eksik_deger_sayisi > 0")
    )
    missing_summary["uygulanan_yontem"] = "Sayısal medyan (Pipeline içinde)"

    y = df[TARGET].map({"No": 0, "Yes": 1}).astype(int)
    X = df.drop(columns=[TARGET, "customerID"])

    service_columns = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    X["ServiceCount"] = X[service_columns].eq("Yes").sum(axis=1)

    tenure_nonzero = X["tenure"].replace(0, np.nan)
    X["AverageMonthlySpend"] = X["TotalCharges"] / tenure_nonzero

    X["TenureGroup"] = pd.cut(
        X["tenure"],
        bins=[-1, 12, 24, 48, np.inf],
        labels=["0-12 ay", "13-24 ay", "25-48 ay", "49+ ay"],
    ).astype("object")

    X["HasSecuritySupport"] = np.where(
        X[["OnlineSecurity", "TechSupport"]].eq("Yes").any(axis=1),
        "Yes",
        "No",
    )

    return X, y, missing_summary


def calculate_outlier_summary(X: pd.DataFrame) -> pd.DataFrame:
    """Sayısal değişkenler için IQR tabanlı aykırı değer sayılarını verir."""
    rows: list[dict[str, Any]] = []
    for column in X.select_dtypes(include=np.number).columns:
        series = X[column].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((series < lower) | (series > upper)).sum())
        rows.append(
            {
                "degisken": column,
                "Q1": q1,
                "Q3": q3,
                "IQR": iqr,
                "alt_sinir": lower,
                "ust_sinir": upper,
                "aykiri_deger_sayisi": count,
            }
        )
    return pd.DataFrame(rows).set_index("degisken")


def split_dataset(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Veriyi stratify ile %60 eğitim, %20 validation ve %20 test ayırır."""
    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_dev,
        y_dev,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_dev,
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def build_pipeline(X: pd.DataFrame, model: Any) -> Pipeline:
    """Doldurma, encoding, ölçekleme, seçim ve modeli tek Pipeline'da kurar."""
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("feature_selection", VarianceThreshold(threshold=0.10)),
            ("model", model),
        ]
    )


def get_models() -> dict[str, Any]:
    """Karşılaştırılacak üç sınıflandırma modelini döndürür."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=15, weights="distance"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    """Modelleri 5 katlı CV ve ayrı validation kümesi üzerinde karşılaştırır."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fitted: dict[str, Pipeline] = {}
    rows: list[dict[str, Any]] = []

    for name, model in get_models().items():
        pipeline = build_pipeline(X_train, model)
        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="f1",
            n_jobs=-1,
        )
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_valid)
        prob = pipeline.predict_proba(X_valid)[:, 1]

        rows.append(
            {
                "model": name,
                "cv_f1_mean": cv_scores.mean(),
                "cv_f1_std": cv_scores.std(),
                "validation_accuracy": accuracy_score(y_valid, pred),
                "validation_precision": precision_score(y_valid, pred, zero_division=0),
                "validation_recall": recall_score(y_valid, pred, zero_division=0),
                "validation_f1": f1_score(y_valid, pred, zero_division=0),
                "validation_roc_auc": roc_auc_score(y_valid, prob),
            }
        )
        fitted[name] = pipeline

    comparison = pd.DataFrame(rows).set_index("model").sort_values(
        "validation_f1", ascending=False
    )
    return comparison, fitted


def get_parameter_grid(model_name: str) -> dict[str, list[Any]]:
    """Validation liderine uygun küçük ve açıklanabilir Grid Search uzayını verir."""
    grids = {
        "Logistic Regression": {
            "model__C": [0.1, 0.5, 1.0, 2.0, 5.0],
            "model__class_weight": [None, "balanced"],
        },
        "KNN": {
            "model__n_neighbors": [7, 11, 15, 21, 31],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
        },
        "Random Forest": {
            "model__n_estimators": [250, 400],
            "model__max_depth": [8, 12, None],
            "model__min_samples_leaf": [1, 4],
            "model__class_weight": [None, "balanced"],
        },
    }
    return grids[model_name]


def tune_best_model(
    model_name: str,
    base_pipeline: Pipeline,
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
) -> GridSearchCV:
    """Seçilen modeli 5 katlı Grid Search ile F1 metriğine göre ayarlar."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        estimator=clone(base_pipeline),
        param_grid=get_parameter_grid(model_name),
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )
    search.fit(X_dev, y_dev)
    return search


def evaluate_on_test(
    estimator: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, float], np.ndarray]:
    """En iyi modeli daha önce görülmemiş test verisinde değerlendirir."""
    pred = estimator.predict(X_test)
    prob = estimator.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, prob),
    }
    return metrics, confusion_matrix(y_test, pred)


def explain_with_permutation(
    estimator: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Modelden bağımsız permutation importance ile özellikleri açıklar."""
    result = permutation_importance(
        estimator,
        X_test,
        y_test,
        scoring="f1",
        n_repeats=8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return (
        pd.DataFrame(
            {
                "feature": X_test.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def save_artifacts(
    output_dir: str | Path,
    comparison: pd.DataFrame,
    search: GridSearchCV,
    test_metrics: dict[str, float],
    cm: np.ndarray,
    importance: pd.DataFrame,
    outlier_summary: pd.DataFrame,
) -> None:
    """Model, metrik, tablo ve grafikleri outputs klasörüne kaydeder."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison.to_csv(output_dir / "model_comparison.csv")
    importance.to_csv(output_dir / "permutation_importance.csv", index=False)
    outlier_summary.to_csv(output_dir / "outlier_summary.csv")
    joblib.dump(search.best_estimator_, output_dir / "best_model.joblib")

    with (output_dir / "best_params.json").open("w", encoding="utf-8") as file:
        json.dump(search.best_params_, file, ensure_ascii=False, indent=2)
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(test_metrics, file, ensure_ascii=False, indent=2)

    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Kalmadı", "Ayrıldı"],
        yticklabels=["Kalmadı", "Ayrıldı"],
    )
    plt.title("Test Verisi - Confusion Matrix")
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    top = importance.head(15).sort_values("importance_mean")
    plt.figure(figsize=(9, 6))
    plt.barh(top["feature"], top["importance_mean"], color="#4C78A8")
    plt.xlabel("Ortalama F1 düşüşü")
    plt.ylabel("Özellik")
    plt.title("Permutation Importance - En Önemli 15 Özellik")
    plt.tight_layout()
    plt.savefig(output_dir / "permutation_importance.png", dpi=150)
    plt.close()


def run_project(project_root: str | Path | None = None) -> dict[str, Any]:
    """Tüm makine öğrenmesi akışını çalıştırır ve sonuçları döndürür."""
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[1]
    data_path = root / "data" / "Telco-Customer-Churn.csv"
    output_dir = root / "outputs"

    raw_df = load_dataset(data_path)
    X, y, missing_summary = engineer_features(raw_df)
    outlier_summary = calculate_outlier_summary(X)
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_dataset(X, y)

    comparison, fitted_models = compare_models(X_train, y_train, X_valid, y_valid)
    best_model_name = comparison.index[0]

    X_dev = pd.concat([X_train, X_valid], axis=0)
    y_dev = pd.concat([y_train, y_valid], axis=0)
    search = tune_best_model(best_model_name, fitted_models[best_model_name], X_dev, y_dev)
    test_metrics, cm = evaluate_on_test(search.best_estimator_, X_test, y_test)
    importance = explain_with_permutation(search.best_estimator_, X_test, y_test)

    save_artifacts(
        output_dir,
        comparison,
        search,
        test_metrics,
        cm,
        importance,
        outlier_summary,
    )

    selected_count = int(
        search.best_estimator_.named_steps["feature_selection"].get_support().sum()
    )
    preprocessed_count = int(
        len(search.best_estimator_.named_steps["preprocessor"].get_feature_names_out())
    )

    return {
        "raw_df": raw_df,
        "X": X,
        "y": y,
        "missing_summary": missing_summary,
        "outlier_summary": outlier_summary,
        "splits": {
            "X_train": X_train,
            "X_valid": X_valid,
            "X_test": X_test,
            "y_train": y_train,
            "y_valid": y_valid,
            "y_test": y_test,
        },
        "comparison": comparison,
        "best_model_name": best_model_name,
        "search": search,
        "test_metrics": test_metrics,
        "confusion_matrix": cm,
        "importance": importance,
        "preprocessed_feature_count": preprocessed_count,
        "selected_feature_count": selected_count,
    }


def main() -> None:
    """Komut satırı giriş noktası."""
    results = run_project()
    print("\nMODEL KARŞILAŞTIRMASI")
    print(results["comparison"].round(4).to_string())
    print(f"\nValidation lideri: {results['best_model_name']}")
    print(f"En iyi Grid Search parametreleri: {results['search'].best_params_}")
    print(f"En iyi CV F1: {results['search'].best_score_:.4f}")
    print("\nTEST METRİKLERİ")
    for name, value in results["test_metrics"].items():
        print(f"{name:>10}: {value:.4f}")
    print("\nÇıktılar outputs/ klasörüne kaydedildi.")


if __name__ == "__main__":
    main()
