import numpy as np


def make_slices(a, n):
    k, m = divmod(a, n)
    return ((i * k + min(i, m), (i + 1) * k + min(i + 1, m)) for i in range(n))


def deepmerge_dicts(source, destination):
    """
    merges source into destination
    """

    for key, value in source.items():
        if isinstance(value, dict):
            # get node or create one
            node = destination.setdefault(key, {})
            deepmerge_dicts(value, node)
        else:
            destination[key] = value

    return destination


def find_closest_time(sample_time, time_base: np.ndarray):
    indices = np.searchsorted(sample_time, time_base, side='right') - 1
    dt = np.median(np.diff(time_base))
    right_mask = time_base-sample_time[indices] > dt/2
    indices[right_mask] = indices[right_mask] + 1
    
    return indices
