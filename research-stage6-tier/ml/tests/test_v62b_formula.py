import numpy as np

from ml.v62b_formula import dense_rank_normalized, v62b_rank_from_columns


def test_dense_rank_normalized_unique():
    x = np.array([0.2, 0.1, 0.3])
    r = dense_rank_normalized(x)
    assert np.allclose(r, np.array([2 / 3, 1 / 3, 1.0]))


def test_dense_rank_normalized_ties():
    x = np.array([1.0, 1.0, 2.0, 5.0, 5.0])
    r = dense_rank_normalized(x)
    # unique values => 3 dense ranks, normalized by 3
    assert np.allclose(r, np.array([1 / 3, 1 / 3, 2 / 3, 1.0, 1.0]))


def test_v62b_rank_from_columns_reproduces_dense_behavior():
    da = np.array([0.0, 0.0, 1.0])
    mix = np.array([0.0, 0.1, 0.0])
    ori = np.array([0.0, 0.0, 0.0])
    rank = v62b_rank_from_columns(da, mix, ori)
    # scores: [0.0, 0.03, 0.6] => dense ranks 1,2,3 => 1/3,2/3,1
    assert np.allclose(rank, np.array([1 / 3, 2 / 3, 1.0]))

