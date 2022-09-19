import sys

from scipy.io import savemat
import numpy as np
import argparse
import time
from datetime import datetime

from multitrackpy import mtt, mvd, helper
from multitrackpy import tracking
from multitrackpy.tracking_opts import get_default_opts


def main():
    print(datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
    # Get default job options
    opts = get_default_opts()

    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Detect positions in multitrack file")
    parser.add_argument('START_IDX', type=int, help="Start frame idx")
    parser.add_argument('END_IDX', type=int, help="End frame idx")
    for key in opts:
        if not key == 'frame_idxs':
            parser.add_argument('--' + key, type=type(opts[key]), required=opts[key] == '', nargs=1)
    args = parser.parse_args()

    # Modify defaults from command line
    for key in opts:
        if not key == 'frame_idxs' and args.__dict__[key] is not None:
            opts[key] = args.__dict__[key][0]

    # Build frame range from command line
    if args.END_IDX == -1:
        args.END_IDX = mtt.read_frame_n(opts['mtt_file'])
    opts['frame_idxs'] = range(args.START_IDX, args.END_IDX)

    # Build video frames
    if not args.mvd_file == 'None':
        vidnames = mtt.read_video_paths(opts['video_dir'], opts['mtt_file'], filenames_only=True)
        time_base = mtt.read_time_base(opts['mtt_file'])
        mvd_times = mvd.read_times(opts['mvd_file'], vidnames)
        opts['frame_maps'] = [helper.find_closest_time(mvd_time, time_base) for mvd_time in mvd_times]

    # Detect frames
    (R, t, errors, fr_out) = tracking.track_frames(opts)

    # Output detection success
    ref_errs = np.max(np.sort(errors, axis=1)[:, 0:3], axis=1)
    print(f'{np.sum(ref_errs < 0.1)}/{len(fr_out)}')

    # Save result together with space_coords
    space_coords = mtt.read_spacecoords(opts['mtt_file'])
    savename = f'{opts["mtt_file"][0:-4]}_pydetect_{fr_out[0] + 1}-{fr_out[-1] + 1}.mat'
    mdic = {'frames': fr_out + 1, 'R': R, 't': t, 'errors': errors, 'space_coords': space_coords}
    savemat(savename, mdic)


if __name__ == "__main__":
    main()
