from brainstat.stats.terms import FixedEffect, MixedEffect
import numpy as np

def test_fixed_intercept():
    random_data = np.random.random_sample((10,1))
    fix1 = FixedEffect(random_data, ['x0'])
    fix2 = FixedEffect(random_data, ['x0'], add_intercept=False)

    assert np.array_equal(fix1.m.shape, [10, 2])
    assert np.array_equal(fix1.names, ['intercept', 'x0'])
    assert np.array_equal(fix2.m.shape, [10, 1])
    assert np.array_equal(fix2.names, ['x0'])


def test_fixed_overload():
    random_data = np.random.random_sample((10,3))
    fix01 = FixedEffect(random_data[:, :2], ['x0', 'x1'], add_intercept=False)
    fix12 = FixedEffect(random_data[:, 1:], ['x2', 'x3'], add_intercept=False)
    fix2 = FixedEffect(random_data[:, 2], ['x2'], add_intercept=False)
    fixi0 = FixedEffect(random_data[:, 0], ['x0'], add_intercept=True)
    fixi1 = FixedEffect(random_data[:, 1], ['x1'], add_intercept=True)

    fix_add = fix01 + fix12
    assert np.array_equal(fix_add.m, random_data)

    fix_add_intercept = fixi0 + fixi1
    expected = np.concatenate((np.ones((10,1)), random_data[:,0:2]), axis=1)
    assert np.array_equal(fix_add_intercept.m, expected)

    fix_sub = fix01 - fix12
    assert np.array_equal(fix_sub.m, random_data[:,0][:, None])

    fix_mul = fix01 * fix2
    assert np.array_equal(fix_mul.m, random_data[:, :2] * random_data[:, 2][:, None])

