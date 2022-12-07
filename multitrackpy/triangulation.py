import imageio
import numpy as np

import calibcamlib
from multitrackpy import mtt, image, pointcloud


def track_frames_sp(opts,
                    space_coords=None, camera_setup: calibcamlib.Camerasystem = None, videos=None, readers=None, offsets=None,
                    R=None, t=None, errors=None, fr_out=None  # E.g. for writing directly into slices of larger array
                    ):
    frame_idxs = opts['frame_idxs']

    # Get inputs if not supplied
    if space_coords is None:
        space_coords = mtt.read_spacecoords(opts['mtt_file'])
    if camera_setup is None:
        calib = mtt.read_calib(opts['mtt_file'])
        camera_setup = calibcamlib.Camerasystem.from_mcl(calib)
    if videos is None:
        videos = mtt.read_video_paths(opts['video_dir'], opts['mtt_file'])
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
        if opts['frame_maps'] is None:
            frames = np.array(
                [image.get_processed_frame(np.double(readers[iC].get_data(fr))) for iC in range(len(videos))])
        else:
            frames = []
            for iC in range(len(videos)):
                frame_idx = opts['frame_maps'][iC][fr]
                if frame_idx<0:
                    frames.append(image.get_processed_frame(np.double(np.zeros(shape=readers[iC].get_data(0).shape))))
                else:
                    frames.append(image.get_processed_frame(np.double(readers[iC].get_data(frame_idx))))

        # print(f'{fr} {time.time()} compute minima')
        minima = [np.flip(image.get_minima(frames[iC], opts['led_thres']), axis=1) for iC in
                  range(len(videos))]  # minima return mat idxs, camera expects xy

        # print(f'{fr} {time.time()} triangulate')
        points = camera_setup.triangulate_nopointcorr(minima, offsets, opts['linedist_thres'], max_points=20)

        fr_out[i] = fr

        # print(f'{fr} {time.time()} find trafo')
        if len(points) > 0:
            R[i], t[i], errors[i] = pointcloud.find_trafo_nocorr(space_coords, points, opts['corr_thres'])
        # print(f'{fr} {time.time()} done')

        if not np.any(np.isnan(R[i])):
            print(f"Found pose in frame {fr} ({[opts['frame_maps'][iC][fr] for iC in range(len(videos))]})")

    return R, t, errors, fr_out
