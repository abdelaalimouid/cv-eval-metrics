"""Two small worked examples that need no model.

1. Precision fooled by one outlier (Naeem et al., ICML 2020), in 2-D:
   real points cluster at the origin, generated points are a bad cluster far
   away. Adding a single real outlier near the bad cluster gives it a huge
   k-NN ball, and precision jumps although the fakes did not change.
2. One Elo update from a class vote between image A and image B.
"""
import numpy as np

from common import save_json


def knn_radii(x, k):
    d = np.linalg.norm(x[:, None] - x[None], axis=-1)
    return np.sort(d, axis=1)[:, k]


def precision(real, fake, k=3):
    d = np.linalg.norm(real[:, None] - fake[None], axis=-1)
    return float((d <= knn_radii(real, k)[:, None]).any(axis=0).mean())


def elo_update(r_a, r_b, a_won, k=32):
    """Return new ratings after one game. Expected score uses the 400-point logistic."""
    e_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    s_a = 1.0 if a_won else 0.0
    return r_a + k * (s_a - e_a), r_b - k * (s_a - e_a), e_a


def main():
    rng = np.random.default_rng(0)
    real = rng.normal(0, 1, (200, 2))
    good = rng.normal(0, 1, (100, 2))            # same distribution as real
    bad = rng.normal((6, 6), 0.5, (100, 2))      # far from every real point
    fakes = np.vstack([good, bad])

    # three real outliers close together, so their k=3 ball is large but they
    # still have neighbours: the k-NN radius reaches back to the main cluster
    outliers = np.array([[5.0, 5.5], [6.5, 5.0], [5.5, 7.0]])
    p_before = precision(real, fakes)
    p_after = precision(np.vstack([real, outliers]), fakes)

    new_a, new_b, expected = elo_update(1000, 1000, a_won=True)
    out = {
        "precision_without_outliers": round(p_before, 3),
        "precision_with_3_real_outliers": round(p_after, 3),
        "elo": {"start": [1000, 1000], "expected_A": expected,
                "after_A_wins": [round(new_a, 1), round(new_b, 1)]},
    }
    save_json("generation_toy", out)
    print(out)


if __name__ == "__main__":
    main()
