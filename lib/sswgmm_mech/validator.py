import numpy as np
from scipy.signal import savgol_filter, find_peaks
from sklearn.mixture import GaussianMixture


def inliers_counter(curve_norm, bound=0.1):
    """
    Count the percentage of inliers
    @param curve_norm: normalized curve
    @param bound: distance from 0 and 1 which a data point can be classified as an inlier
    @return:
    percent: the percentage of inliers
    """
    inliers_0 = np.where(np.abs(curve_norm - 1) <= bound)[0]
    inliers_1 = np.where(np.abs(curve_norm - 0) <= bound)[0]
    inliers_0_percent = len(inliers_0) / len(curve_norm)
    inliers_1_percent = len(inliers_1) / len(curve_norm)
    percent = inliers_0_percent + inliers_1_percent
    percent = np.clip(percent, 0, 1)
    if (inliers_0_percent > 0.99) or (inliers_1_percent > 0.99):
        percent = 0
    return percent


def current_prescreen(curve, inliers_percent=0.8):
    """
    Check current data quality. Detects abnormal curve shape and data with very high noise level.
    @param curve: zeroed open or close current data
    @param inliers_percent: the percentage of inliers required to be classified as good data. Default is 0.8
    @return:
    percent: Boolean, whether the data is bad in terms of inliers
    noise_level: Boolean, whether the data is bad in terms of noise level
    """
    # noise level estimation
    curve_smooth = savgol_filter(curve, 51, 2)
    noise = np.abs(curve_smooth - curve)
    noise_level = (noise > (curve.max() - curve.min()) / 4).any()

    # inlier counter
    curve_norm = (curve - np.mean(curve[:100])) / curve.max()
    percent = inliers_counter(curve_norm)
    # minimum requirement for current change
    diff = curve.max() - curve.min()
    if (percent < inliers_percent) or (diff < 100):
        percent = True
    else:
        percent = False
    return percent, noise_level


def angle_prescreen(curve, inliers_percent=0.8):
    """
    Check angle data quality. Detects abnormal curve shape and data with very high noise level.
    @param curve: raw angle sensor data, np array of shape (-1,)
    @param inliers_percent: the percentage of inliers required to be classified as good data. Default is 0.8
    @return:
    percent: Boolean, whether the data is bad in terms of inliers
    invalid_curve: Boolean, whether the data is invalid in terms the number of turning points
    splits: index for where to split a curve into multiple curves. np array of shape (-1,)
    head: upper 'soft' bound of the curve
    tail: lower 'soft' bound of the curve
    """
    curve = np.array(curve)
    gm = GaussianMixture(3, covariance_type='spherical', n_init=1)
    gm.fit(curve.reshape(-1, 1))
    ix = np.argsort(gm.covariances_)[:-1].reshape(-1)
    tail, head = np.sort(gm.means_.reshape(-1)[ix])
    center = (head + tail) / 2

    # normalize curve
    curve_norm = (curve - tail) / (head - tail)

    # count inliers
    percent = inliers_counter(curve_norm)
    if percent < inliers_percent:
        percent = True  # True for problematic signal
    else:
        percent = False

    # split from center
    ix = np.where(curve > center)[0]
    curve[ix] = center + (center - curve[ix])

    # count splits
    height = (head - tail) / 2.1
    splits, _ = find_peaks(curve - tail, height=height, distance=100)
    if len(splits) in [1, 2, 3]:
        invalid_curve = False
    else:
        invalid_curve = True
    return percent, invalid_curve, splits
