import numpy as np
from ccvtools import rawio  # noqa
from multiprocessing import Pool
from functools import partial
import time

import calibcamlib
from multitrackpy import mtt
from multitrackpy import helper
from multitrackpy.triangulation import track_frames_sp


def track_frames(opts):
    print('Running deciding mode')
    if opts['n_cpu'] > 1:
        print('Running in MP mode')
        return detect_frames_mp(opts)
    else:
        print('Running in SP mode')
        return track_frames_sp(opts)


def detect_frames_mp(opts):
    print(f'Tracking started in MP mode / {time.time()}')
    space_coords = mtt.read_spacecoords(opts['mtt_file'])
    calib = mtt.read_calib(opts['mtt_file'])
    videos = mtt.read_video_paths(opts['video_dir'], opts['mtt_file'])
    print(f'Using {len(videos)} tracking cams')

    camera_setup = calibcamlib.Camerasystem.from_mcl(calib)

    preloaddict = {
        'space_coords': space_coords,
        'camera_setup':  camera_setup,
        'videos': videos,
    }

    frame_idxs = np.asarray(list(opts['frame_idxs']))
    R = np.empty((len(frame_idxs), 3, 3))
    R[:] = np.NaN
    t = np.empty((len(frame_idxs), 3, 1))
    t[:] = np.NaN
    errors = np.empty((len(frame_idxs), space_coords.shape[0]))
    errors[:] = np.NaN
    fr_out = np.empty((len(frame_idxs)), dtype=np.int32)

    slice_list = list(helper.make_slices(len(frame_idxs), opts['n_cpu']))
    # shallow-merge frame ranges into copies of opts
    arg_list = [opts.copy() | {'frame_idxs': frame_idxs[sl[0]:sl[1]]} for sl in slice_list]

    print(f'Using {opts["n_cpu"]} workers')
    with Pool(opts['n_cpu']) as p:
        pres_list = p.map(partial(track_frames_sp, **preloaddict), arg_list)

    for (sl, pres) in zip(slice_list, pres_list):  # Poolmap() returns in order
        R[sl[0]:sl[1]] = pres[0]
        t[sl[0]:sl[1]] = pres[1]
        errors[sl[0]:sl[1]] = pres[2]
        fr_out[sl[0]:sl[1]] = pres[3]

    return R, t, errors, fr_out
