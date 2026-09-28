import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    accuracy_score, precision_score, recall_score
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_SETS = {
    "input_only": ["question_chars", "question_words"],
    "text_proxies": [
        "response_chars", "response_words", "sentence_count", "numeric_token_count",
        "hedge_count", "reversal_count", "contradiction_phrase_count",
        "distinct_numeric_values"
    ],
    "agreement_only": ["agreement_fraction", "unique_answer_count", "valid_answer_fraction"],
    "combined": [
        "question_chars", "question_words", "response_chars", "response_words",
        "sentence_count", "numeric_token_count", "hedge_count", "reversal_count",
        "contradiction_phrase_count", "distinct_numeric_values",
        "agreement_fraction", "unique_answer_count", "valid_answer_fraction"
    ],
}

def matrix(rows, feature_names):
    return np.asarray([[float(r.get(name, 0.0)) for name in feature_names] for r in rows], dtype=float)

def fit_model(train_rows, feature_names):
    X = matrix(train_rows, feature_names)
    y = np.asarray([int(r["utility_positive"]) for r in train_rows])
    if len(np.unique(y)) < 2:
        return None
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=3000, random_state=17)
    )
    model.fit(X, y)
    return model

def probabilities(model, rows, feature_names):
    return model.predict_proba(matrix(rows, feature_names))[:, 1]

def metric_report(y, p, threshold=0.5):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    result = {
        "n": int(len(y)),
        "positive_rate": float(y.mean()) if len(y) else None,
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)) if len(y) else None,
        "precision": float(precision_score(y, pred, zero_division=0)) if len(y) else None,
        "recall": float(recall_score(y, pred, zero_division=0)) if len(y) else None,
        "brier": float(brier_score_loss(y, p)) if len(y) else None,
    }
    result["auroc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None
    result["auprc"] = float(average_precision_score(y, p)) if len(y) else None
    return result

def choose_threshold(y, p):
    """Choose threshold on validation set by maximum F1; ties favor larger threshold."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    candidates = sorted(set([0.0, 0.5, 1.0] + p.tolist()))
    best = (float("-inf"), 0.5)
    for t in candidates:
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom else 0.0
        candidate = (f1, t)
        if candidate > best:
            best = candidate
    return float(best[1])

def policy_report(rows, predicted_continue):
    """Compute final accuracy and token use for stop vs add-two-samples policy."""
    correct, tokens, latency, count = [], [], [], []
    for row, cont in zip(rows, predicted_continue):
        if cont:
            correct.append(row["post_action_correct"])
            tokens.append(row["total_output_tokens"])
            latency.append(row["total_latency_seconds"])
            count.append(5)
        else:
            correct.append(row["baseline_correct"])
            tokens.append(row["initial_output_tokens"])
            latency.append(row["initial_latency_seconds"])
            count.append(3)
    n = max(len(rows), 1)
    return {
        "accuracy": float(sum(correct) / n),
        "mean_output_tokens": float(sum(tokens) / n),
        "mean_latency_seconds": float(sum(latency) / n),
        "mean_samples": float(sum(count) / n),
        "continued_fraction": float(sum(bool(x) for x in predicted_continue) / n),
    }


def split_rows(rows, seed, train_fraction=0.6, validation_fraction=0.2):
    """Problem-level split; each row represents one unique problem."""
    import random
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("Fractions must be positive and sum to less than 1.")
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    n_train = int(len(indices) * train_fraction)
    n_val = int(len(indices) * validation_fraction)
    train_i = set(indices[:n_train])
    val_i = set(indices[n_train:n_train+n_val])
    test_i = set(indices[n_train+n_val:])
    return (
        [r for i, r in enumerate(rows) if i in train_i],
        [r for i, r in enumerate(rows) if i in val_i],
        [r for i, r in enumerate(rows) if i in test_i],
    )
