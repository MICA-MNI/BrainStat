""" Tests the fixed and mixed effects classes. """

import numpy as np

from brainstat.stats.terms import FixedEffect, MixedEffect


def test_fixed_init():
    """Tests the initialization of the FixedEffect class."""
    random_data = np.random.random_sample((10, 1))
    fix1 = FixedEffect(random_data, ["x0"])
    fix2 = FixedEffect(random_data, ["x0"], add_intercept=False)

    assert np.array_equal(fix1.m.shape, [10, 2])
    assert np.array_equal(fix1.names, ["intercept", "x0"])

    assert np.array_equal(fix2.m.shape, [10, 1])
    assert np.array_equal(fix2.names, ["x0"])


def test_fixed_overload():
    """Tests the overloads of the FixedEffect class."""
    random_data = np.random.random_sample((10, 3))
    fix01 = FixedEffect(random_data[:, :2], ["x0", "x1"], add_intercept=False)
    fix12 = FixedEffect(random_data[:, 1:], ["x2", "x3"], add_intercept=False)
    fix2 = FixedEffect(random_data[:, 2], ["x2"], add_intercept=False)
    fixi0 = FixedEffect(random_data[:, 0], ["x0"], add_intercept=True)
    fixi1 = FixedEffect(random_data[:, 1], ["x1"], add_intercept=True)

    fix_add = fix01 + fix12
    assert np.array_equal(fix_add.m, random_data)

    fix_add_intercept = 1 + FixedEffect(random_data[:, 0])
    assert np.array_equal(fixi0.m, fix_add_intercept.m)

    fix_add_intercept = fixi0 + fixi1
    expected = np.concatenate((np.ones((10, 1)), random_data[:, 0:2]), axis=1)
    assert np.array_equal(fix_add_intercept.m, expected)

    fix_sub = fix01 - fix12
    assert np.array_equal(fix_sub.m, random_data[:, 0][:, None])

    fix_mul = fix01 * fix2
    assert np.array_equal(fix_mul.m, random_data[:, :2] * random_data[:, 2][:, None])


def test_mixed_init():
    """Tests the initialization of the MixedEffect class."""
    n = 10
    random_data = np.random.random_sample((n, 1))
    mix1 = MixedEffect(random_data, ["x0"])
    mix2 = MixedEffect(random_data, ["x0"], add_identity=False)
    mix3 = MixedEffect(random_data, random_data, ["x0"], ["y0"])
    mix4 = MixedEffect(random_data, random_data, ["x0"], ["y0"], add_intercept=False)

    assert np.array_equal(mix1.variance.shape, [n ** 2, 2])
    assert np.array_equal(mix1.variance.names, ["x0", "I"])

    assert np.array_equal(mix2.variance.shape, [n ** 2, 1])
    assert np.array_equal(mix2.variance.names, ["x0"])

    assert np.array_equal(mix3.mean.shape, [10, 2])
    assert np.array_equal(mix3.mean.names, ["intercept", "y0"])

    assert np.array_equal(mix4.mean.shape, [10, 1])
    assert np.array_equal(mix4.mean.names, ["y0"])


def test_mixed_overload():
    """Tests the overloads of the MixedEffect class."""
    n = 3
    random_data = np.random.random_sample((n, 4))
    mix1 = MixedEffect(random_data[:, 0], name_ran=["x0"])
    mix2 = MixedEffect(random_data[:, 1], name_ran=["x1"])

    I = np.identity(n).flatten()[:, None]
    var12 = as_variance(random_data[:, :2])

    mix_add = mix1 + mix2
    expected_add = np.concatenate((var12, I), axis=1)
    assert np.array_equal(mix_add.variance.m, expected_add)

    mix_sub = mix1 - mix1
    assert mix_sub.empty

    mix_mul = mix1 * mix2
    expected_mul = np.concatenate(
        (
            var12[:, 0][:, None] * var12[:, 1][:, None],
            var12[:, 1][:, None] * I,
            var12[:, 0][:, None] * I,
            I,
        ),
        axis=1,
    )
    assert np.array_equal(mix_mul.variance.m, expected_mul)


def test_identity_detection():
    """Tests that the identity matrix is correctly placed last."""
    mix1 = MixedEffect(np.random.rand(3, 1), add_identity=False)
    mix2 = MixedEffect(1, name_ran="test_identity")
    I = np.identity(3).flatten()

    mix_add1 = mix2 + mix1
    mix_add2 = mix1 + mix2
    assert np.all(mix_add1.variance.m.to_numpy()[:, 1] == I)
    assert np.all(mix_add2.variance.m.to_numpy()[:, 1] == I)


def as_variance(M):
    var = [np.reshape(x[:, None] @ x[None, :], (-1, 1)) for x in M.T]
    return np.squeeze(var, axis=2).T
