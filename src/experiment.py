"""Reproducible baseline-versus-MLP experiment on sklearn digits."""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001
RESULTS_DIR = Path("results")


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(SEED)


def load_data():
    digits = load_digits()
    X = digits.images.reshape(len(digits.images), -1).astype(np.float32) / 16.0
    y = digits.target.astype(np.int64)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.20,
        random_state=SEED,
        stratify=y_trainval,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


class SmallMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.network(x)


def classification_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def predict_mlp(model, X, device):
    model.eval()
    with torch.no_grad():
        inputs = torch.as_tensor(X, dtype=torch.float32, device=device)
        return model(inputs).argmax(dim=1).cpu().numpy()


def main():
    set_seed()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    # Baseline: standardised pixel features and logistic regression.
    baseline = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=SEED),
    )
    start = time.perf_counter()
    baseline.fit(X_train, y_train)
    baseline_seconds = time.perf_counter() - start

    baseline_val_pred = baseline.predict(X_val)
    baseline_val = classification_metrics(y_val, baseline_val_pred)

    # Neural network.
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    train_ds = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    model = SmallMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    best_val_accuracy = -1.0
    best_state = None
    start = time.perf_counter()

    for epoch in range(EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(inputs), labels)
            loss.backward()
            optimizer.step()

        val_pred = predict_mlp(model, X_val, device)
        val_metrics = classification_metrics(y_val, val_pred)

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS}: {val_metrics}")

    mlp_seconds = time.perf_counter() - start
    model.load_state_dict(best_state)

    mlp_val_pred = predict_mlp(model, X_val, device)
    mlp_val = classification_metrics(y_val, mlp_val_pred)

    # Final test evaluation after model selection.
    baseline_test_pred = baseline.predict(X_test)
    mlp_test_pred = predict_mlp(model, X_test, device)



    # Robustness evaluation using the same noise pattern at each severity.
    rng = np.random.default_rng(SEED)
    standard_noise = rng.normal(0.0, 1.0, size=X_test.shape)
    robustness_results = []

    for noise_std in (0.0, 0.05, 0.10, 0.20, 0.30):
        noise = noise_std * standard_noise
        X_corrupted = np.clip(
            X_test + noise, 0.0, 1.0
        ).astype(np.float32)

        baseline_corrupted_pred = baseline.predict(X_corrupted)
        mlp_corrupted_pred = predict_mlp(model, X_corrupted, device)

        robustness_results.append({
            "noise_std": noise_std,
            "baseline": classification_metrics(
                y_test, baseline_corrupted_pred
            ),
            "mlp": classification_metrics(
                y_test, mlp_corrupted_pred
            ),
        })

    with (RESULTS_DIR / "robustness.json").open("w") as f:
        json.dump(robustness_results, f, indent=2)


    with (RESULTS_DIR / "robustness.json").open("w") as f:
        json.dump(robustness_results, f, indent=2)


    results = {
        "dataset": "scikit-learn digits",
        "samples": 1797,
        "image_dimensions": [8, 8],
        "split_sizes": {
            "train": len(X_train),
            "validation": len(X_val),
            "test": len(X_test),
        },
        "seed": SEED,
        "device": str(device),
        "settings": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
        },
        "baseline": {
            "validation": baseline_val,
            "test": classification_metrics(y_test, baseline_test_pred),
            "training_seconds": baseline_seconds,
        },
        "mlp": {
            "validation": mlp_val,
            "test": classification_metrics(y_test, mlp_test_pred),
            "training_seconds": mlp_seconds,
        },
    }

    with (RESULTS_DIR / "metrics.json").open("w") as f:
        json.dump(results, f, indent=2)

    # Save per-image predictions so error analysis matches this exact run.
    predictions = np.column_stack(
        [
            np.arange(len(y_test)),
            y_test,
            baseline_test_pred,
            mlp_test_pred,
            baseline_test_pred != y_test,
            mlp_test_pred != y_test,
        ]
    )

    np.savetxt(
        RESULTS_DIR / "test_predictions.csv",
        predictions,
        delimiter=",",
        fmt="%d",
        header="test_index,true_label,baseline_prediction,mlp_prediction,baseline_wrong,mlp_wrong",
        comments="",
    )

    error_analysis = {
        "test_samples": int(len(y_test)),
        "baseline_errors": int(np.sum(baseline_test_pred != y_test)),
        "mlp_errors": int(np.sum(mlp_test_pred != y_test)),
        "both_wrong": int(
            np.sum(
                (baseline_test_pred != y_test)
                & (mlp_test_pred != y_test)
            )
        ),
        "baseline_only_wrong": int(
            np.sum(
                (baseline_test_pred != y_test)
                & (mlp_test_pred == y_test)
            )
        ),
        "mlp_only_wrong": int(
            np.sum(
                (baseline_test_pred == y_test)
                & (mlp_test_pred != y_test)
            )
        ),
        "predictions_disagree": int(
            np.sum(baseline_test_pred != mlp_test_pred)
        ),
    }

    with (RESULTS_DIR / "error_analysis.json").open("w") as f:
        json.dump(error_analysis, f, indent=2)


    np.savetxt(
        RESULTS_DIR / "baseline_confusion_matrix.csv",
        confusion_matrix(y_test, baseline_test_pred),
        delimiter=",",
        fmt="%d",
    )
    np.savetxt(
        RESULTS_DIR / "mlp_confusion_matrix.csv",
        confusion_matrix(y_test, mlp_test_pred),
        delimiter=",",
        fmt="%d",
    )

    print("\nFinal results")
    print(json.dumps(results, indent=2))
    print("\nMLP test classification report")
    print(classification_report(y_test, mlp_test_pred, digits=4))
    print("\nSaved results to:", RESULTS_DIR.resolve())


if __name__ == "__main__":
    main()
