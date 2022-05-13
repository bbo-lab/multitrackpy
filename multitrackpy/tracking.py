import numpy as np
import imageio
from ccvtools import rawio  # noqa
from multiprocessing import Pool
from functools import partial
import time

import calibcamlib
from . import mtt
from . import image
from . import pointcloud
from . import helper


def track_frames(gopts):
    print('Running deciding mode')
    if gopts['n_cpu'] > 1:
        print('Running in MP mode')
        return track_frames_mp(gopts)
    else:
        print('Running in SP mode')
        return track_frames_sp(gopts)


def track_frames_sp(gopts,
                    space_coords=None, camera_setup: calibcamlib.Camerasystem = None, videos=None, readers=None, offsets=None,
                    R=None, t=None, errors=None, fr_out=None  # E.g. for writing directly into slices of larger array
                    ):
    frame_idxs = gopts['frame_idxs']

    # Get inputs if not supplied
    if space_coords is None:
        space_coords = mtt.read_spacecoords(gopts['mtt_file'])
    if camera_setup is None:
        calib = mtt.read_calib(gopts['mtt_file'])
        camera_setup = calibcamlib.Camerasystem.from_mcl(calib)
    if videos is None:
        videos = mtt.read_video_paths(gopts['video_dir'], gopts['mtt_file'])
    if readers is None:
        readers = [imageio.get_reader(videos[i]) for i in range(len(videos))]
    if offsets is None:
        offsets = np.array([reader.header['sensor']['offset'] for reader in readers])

    if R is None:
        assert (t is None and errors is None and fr_out is None)
        R = np.empty((len(frame_idxs), 3, 3))
        t = np.empty((len(frame_idxs), 3, 1))
        errors = np.empty((len(frame_idxs), space_coords.shape[0]))
        fr_out = np.empty((len(frame_idxs)), dtype=np.int32)
    else:
        assert (t is not None and errors is not None and fr_out is not None)

    # Initilize arrays
    R[:] = np.NaN
    t[:] = np.NaN
    errors[:] = np.NaN

    # Iterate frames for processing
    for (i, fr) in enumerate(frame_idxs):
        # print(f'{fr} {time.time()} fetch data')
        frames = np.array([image.get_processed_frame(np.double(readers[iC].get_data(fr))) for iC in range(len(videos))])

        # print(f'{fr} {time.time()} compute minima')
        minima = [np.flip(image.get_minima(frames[iC], gopts['led_thres']), axis=1) for iC in
                  range(len(videos))]  # minima return mat idxs, camera expects xy

        # print(f'{fr} {time.time()} triangulate')
        points = camera_setup.triangulate_nopointcorr(minima, offsets, gopts['linedist_thres'])

        fr_out[i] = fr

        # print(f'{fr} {time.time()} find trafo')
        if len(points) > 0:
            R[i], t[i], errors[i] = pointcloud.find_trafo_nocorr(space_coords, points, gopts['corr_thres'])
        # print(f'{fr} {time.time()} done')

    return R, t, errors, fr_out


def track_frames_mp(gopts):
    print(f'Tracking started in MP mode / {time.time()}')
    space_coords = mtt.read_spacecoords(gopts['mtt_file'])
    calib = mtt.read_calib(gopts['mtt_file'])
    videos = mtt.read_video_paths(gopts['video_dir'], gopts['mtt_file'])
    print(f'Using {len(videos)} tracking cams')

    camera_setup = calibcamlib.Camerasystem.from_mcl(calib)

    preloaddict = {
        'space_coords': space_coords,
        'camera_setup':  camera_setup,
        'videos': videos,
    }

    frame_idxs = np.asarray(list(gopts['frame_idxs']))
    R = np.empty((len(frame_idxs), 3, 3))
    R[:] = np.NaN
    t = np.empty((len(frame_idxs), 3, 1))
    t[:] = np.NaN
    errors = np.empty((len(frame_idxs), space_coords.shape[0]))
    errors[:] = np.NaN
    fr_out = np.empty((len(frame_idxs)), dtype=np.int32)

    slice_list = list(helper.make_slices(len(frame_idxs), gopts['n_cpu']))
    # shallow-merge frame ranges into copies of gopts
    arg_list = [gopts.copy() | {'frame_idxs': frame_idxs[sl[0]:sl[1]]} for sl in slice_list]

    print(f'Using {gopts["n_cpu"]} workers')
    with Pool(gopts['n_cpu']) as p:
        pres_list = p.map(partial(track_frames_sp, **preloaddict), arg_list)

    for (sl, pres) in zip(slice_list, pres_list):  # Poolmap() returns in order
        R[sl[0]:sl[1]] = pres[0]
        t[sl[0]:sl[1]] = pres[1]
        errors[sl[0]:sl[1]] = pres[2]
        fr_out[sl[0]:sl[1]] = pres[3]

    return R, t, errors, fr_out


def get_default_globalopts():
    return {
        'mtt_file': '',
        'video_dir': '',
        'frame_idxs': None,
        'linedist_thres': 0.2,
        # Max distance between two cam lines to assume that the respective detections come from the same LED (calibration units)
        'corr_thres': 0.1,
        # Max diffeerence in point distance for correlation between model and detection (calibration units)
        'led_thres': 150,  # Minimal brightness of LED after image processing (image brightness units)
        'n_cpu': 2,
    }


def check_globalopts(gopts):
    return (all(name in get_default_globalopts() for name in gopts) and
            all(name in gopts for name in get_default_globalopts()))
