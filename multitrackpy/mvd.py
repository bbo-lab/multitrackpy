import numpy as np
# import h5py
import scipy.io as sio


def read_times(mvd_path, vidnames):
    mvd_file = sio.loadmat(mvd_path)
    # mvd_file = h5py.File(mvd_path)

    times = []
    for name in vidnames:
        videomask = np.array([m['vidname'][0] == name for m in mvd_file['vid'][0]])
        assert videomask.sum() == 1, f"Found {videomask.sum()} matches of video name {name}. Wrond mvd file?"
        times.append(np.asarray(mvd_file['vid'][0][videomask][0]['times']).ravel())

    return times
