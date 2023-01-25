import imageio
import numpy as np
from pathlib import Path

import calibcamlib
from multitrackpy import mtt, image, pointcloud


# noinspection PyPackageRequirements,DuplicatedCode
def track_frames_sp(opts,
                    space_coords=None, camera_setup: calibcamlib.Camerasystem = None, videos=None, readers=None,
                    offsets=None,
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
            cam_fr_idxs = [fr for _ in range(len(videos))]
        else:
            cam_fr_idxs = [opts['frame_maps'][iC][fr] if
                           opts['frame_maps'][iC][fr] >= 0 else 0 for iC in range(len(videos))]

        frames = np.array(
            [image.get_processed_frame(np.double(readers[iC].get_data(cam_fr_idxs[iC]))) for iC in range(len(videos))])

        # print(f'{fr} {time.time()} compute minima')
        minima = [np.flip(image.get_minima(frames[iC], opts['led_thres']), axis=1) for iC in
                  range(len(videos))]  # minima return mat idxs, camera expects xy

        points = camera_setup.triangulate_nopointcorr(minima, offsets, opts['linedist_thres'], max_points=20)

        if opts["debug"] > 0:
            print(f"{fr} ({cam_fr_idxs}): Found {[m.shape[0] for m in minima]} minima.")
            print(f"{fr} ({cam_fr_idxs}): Triangulated {len(points)} points.")
        if opts["debug"] > 1:
            try:
                import matplotlib
                from matplotlib import pyplot as plt
                matplotlib.use('Agg')
                rep_points = camera_setup.project(points, offsets)
                for i_cam, frame in enumerate(frames):
                    fig = plt.figure(frameon=False, figsize=(frame.shape[0]/16, frame.shape[1]/16), dpi=80)
                    ax = plt.Axes(fig, [0., 0., 1., 1.])
                    ax.set_axis_off()
                    fig.add_axes(ax)
                    plt.imshow(frame)
                    plt.plot(minima[i_cam][:, 0], minima[i_cam][:, 1], 'r+')
                    plt.savefig(Path(opts["mtt_file"]).parent / f"debug_minima_{i_cam}_{fr}_{cam_fr_idxs[i_cam]}.png")

                    fig = plt.figure(frameon=False, figsize=(frame.shape[0]/16, frame.shape[1]/16), dpi=80)
                    ax = plt.Axes(fig, [0., 0., 1., 1.])
                    ax.set_axis_off()
                    fig.add_axes(ax)
                    plt.imshow(frame)
                    plt.plot(rep_points[i_cam, :, 0], rep_points[i_cam, :, 1], 'r+')
                    plt.savefig(Path(opts["mtt_file"]).parent / f"debug_triangulation_{i_cam}_{fr}_{cam_fr_idxs[i_cam]}.png")
            except ModuleNotFoundError:
                print(f"Missing module matplotlib, skipping debug image generation.")

        fr_out[i] = fr

        # print(f'{fr} {time.time()} find trafo')
        if len(points) > 0:
            R[i], t[i], errors[i] = pointcloud.find_trafo_nocorr(space_coords, points, opts['corr_thres'], opts['track_ambiguous'])
        # print(f'{fr} {time.time()} done')

        if not np.any(np.isnan(R[i])):
            print(f"{fr} ({cam_fr_idxs}): Found pose")

    return R, t, errors, fr_out
