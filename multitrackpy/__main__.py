import sys
sys.path.insert(0, '/home/voit/code/calibcamlib')

from scipy.io import savemat
import numpy as np
import argparse
import time
from datetime import datetime

from multitrackpy import mtt
from multitrackpy import tracking

def main():
    print(datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
    # Get default job options
    gopts = tracking.get_default_globalopts()

    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Detect positions in multitrack file")
    parser.add_argument('START_IDX', type=int, help="Start frame idx")
    parser.add_argument('END_IDX', type=int, help="End frame idx")
    for key in gopts:
        if not key == 'frame_idxs':
            parser.add_argument('--' + key, type=type(gopts[key]), required=gopts[key] == '', nargs=1)
    args = parser.parse_args()

    # Modify defaults from command line
    for key in gopts:
        if not key == 'frame_idxs' and args.__dict__[key] is not None:
            gopts[key] = args.__dict__[key][0]

    # Build frame range from command line
    if args.END_IDX == -1:
        args.END_IDX = mtt.read_frame_n(gopts['mtt_file'])
    gopts['frame_idxs'] = range(args.START_IDX, args.END_IDX)

    # Detect frames
    (R, t, errors, fr_out) = tracking.track_frames(gopts)

    # Output detection success
    ref_errs = np.max(np.sort(errors, axis=1)[:, 0:3], axis=1)
    print(f'{np.sum(ref_errs < 0.1)}/{len(fr_out)}')

    # Save result together with space_coords
    space_coords = mtt.read_spacecoords(gopts['mtt_file'])
    savename = f'{gopts["mtt_file"][0:-4]}_pydetect_{fr_out[0] + 1}-{fr_out[-1] + 1}.mat'
    mdic = {'frames': fr_out + 1, 'R': R, 't': t, 'errors': errors, 'space_coords': space_coords}
    savemat(savename, mdic)


if __name__ == "__main__":
    main()
