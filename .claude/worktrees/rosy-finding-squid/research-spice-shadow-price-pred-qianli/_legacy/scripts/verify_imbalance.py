import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def test_imbalance_metrics():
    print("Simulating High Imbalance (98% Zeros)...")
    np.random.seed(42)
    n_samples = 10000
    n_pos = 200  # 2% positive labels

    # 1. Create Labels
    # y = 0 for most, y > 0 for some (Gamma distribution for prices)
    y = np.zeros(n_samples)
    y[:n_pos] = np.random.gamma(shape=2, scale=10, size=n_pos) + 10  # Prices ~ 10-50

    y_binary = (y > 0).astype(int)

    print("\nMetric Comparison (Positive Signal):")
    print(f"{'Method':<20} {'Value':<10} {'Interpretation':<20}")
    print("-" * 50)

    # 2. Create a Positive Feature (Weak signal)
    # Background noise
    x = np.random.normal(0, 1, n_samples)
    # Shift x for positive class
    x[:n_pos] += 1.0

    p_corr = np.corrcoef(x, y)[0, 1]
    s_corr, _ = spearmanr(x, y)
    auc = roc_auc_score(y_binary, x)

    print(f"{'Pearson Corr':<20} {p_corr:.4f}     {'Correct Sign (+)' if p_corr > 0 else 'WRONG SIGN'}")
    print(f"{'Spearman Corr':<20} {s_corr:.4f}     {'Correct Sign (+)' if s_corr > 0 else 'WRONG SIGN'}")
    print(f"{'AUC Score':<20} {auc:.4f}     {'Correct range (>0.5)' if auc > 0.5 else 'WRONG RANGE'}")

    print("\nMetric Comparison (No Signal - Noise):")
    x_noise = np.random.normal(0, 1, n_samples)
    p_corr = np.corrcoef(x_noise, y)[0, 1]
    s_corr, _ = spearmanr(x_noise, y)
    auc = roc_auc_score(y_binary, x_noise)
    print(f"{'Pearson Corr':<20} {p_corr:.4f}")
    print(f"{'Spearman Corr':<20} {s_corr:.4f}")
    print(f"{'AUC Score':<20} {auc:.4f}")

    print("\nMetric Comparison (Outlier Effect):")
    # One huge outlier in x for a ZERO label that might flip pearson?
    x_outlier = x.copy()
    x_outlier[-1] = 1000  # Huge value for a label=0 point

    p_corr = np.corrcoef(x_outlier, y)[0, 1]
    s_corr, _ = spearmanr(x_outlier, y)
    auc = roc_auc_score(y_binary, x_outlier)

    print(f"{'Pearson Corr':<20} {p_corr:.4f}     {'Failed' if p_corr < 0 else 'Stable'}")
    print(f"{'Spearman Corr':<20} {s_corr:.4f}     {'Stable' if s_corr > 0 else 'Failed'}")
    print(f"{'AUC Score':<20} {auc:.4f}     {'Stable' if auc > 0.5 else 'Failed'}")


if __name__ == "__main__":
    test_imbalance_metrics()
