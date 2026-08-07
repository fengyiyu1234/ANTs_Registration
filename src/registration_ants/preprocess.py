"""Pre-registration preprocessing: bias correction and intensity normalization."""
import ants
import numpy as np


def n4_correct(img):
    """Correct smooth illumination gradients via N4 bias field correction."""
    return ants.n4_bias_field_correction(img)


def clip_and_normalize(img, lower_pct=0.5, upper_pct=99.5):
    """Percentile-clip intensities and rescale to [0, 1].

    Prevents a handful of outlier-bright voxels from dominating the MI/CC
    cost function during registration.
    """
    arr = img.numpy()
    lo, hi = np.percentile(arr, [lower_pct, upper_pct])
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / max(hi - lo, 1e-8)
    return ants.from_numpy(arr.astype(np.float32), spacing=img.spacing, origin=img.origin, direction=img.direction)


def preprocess_for_registration(img, lower_pct=0.5, upper_pct=99.5):
    """Standard preprocessing chain: N4 bias correction then intensity normalization."""
    return clip_and_normalize(n4_correct(img), lower_pct, upper_pct)
