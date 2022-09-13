import numpy as np
import h5py
import os.path


def read_calib(mtt_path):
    mtt_file = h5py.File(mtt_path)

    istracking = np.squeeze(np.asarray([mtt_file['mt']['cam_istracking']]) == 1)
    calind = np.squeeze(np.int32(mtt_file['mt']['calind']))[istracking] - 1

    mc = {
        'Rglobal': np.asarray(mtt_file['mt']['mc']['Rglobal']).transpose((0, 2, 1))[calind],  # in reverse order in h5 file!
        'Tglobal': np.asarray(mtt_file['mt']['mc']['Tglobal'])[calind],
        'cal': []
    }

    for ci in calind:
        mc['cal'].append({
            'scaling': np.asarray(mtt_file[mtt_file['mt']['mc']['cal']['scaling'][ci, 0]]).ravel(),
            'icent': np.asarray(mtt_file[mtt_file['mt']['mc']['cal']['icent'][ci, 0]]).ravel(),
            'distortion_coefs': np.asarray(mtt_file[mtt_file['mt']['mc']['cal']['distortion_coefs'][ci, 0]]).ravel(),
            'sensorsize': np.asarray(mtt_file[mtt_file['mt']['mc']['cal']['sensorsize'][ci, 0]]).ravel(),
            'scale_pixels': np.asarray(mtt_file[mtt_file['mt']['mc']['cal']['scale_pixels'][ci, 0]]).ravel()[0],
        })

    # pprint(mc)
    return mc


def read_video_paths(vid_dir, mtt_path, filenames_only=False):
    mtt_file = h5py.File(mtt_path)
    istracking = np.squeeze(np.asarray([mtt_file['mt']['cam_istracking']]) == 1)
    filenames = [''.join([chr(c) for c in mtt_file[mtt_file['mt']['vidname'][0, i]][:].T.astype(np.int)[0]]) for i
                 in np.where(istracking)[0]]
    if not filenames_only:
        filenames = [os.path.join(vid_dir, f) for f in filenames]

    return filenames


def read_spacecoords(mtt_path):
    mtt_file = h5py.File(mtt_path)
    return np.asarray(mtt_file['mt']['objmodel']['space_coord'])


def read_frame_n(mtt_path):
    mtt_file = h5py.File(mtt_path)
    return len(mtt_file['mt']['t'])

def read_time_base(mtt_file):
    mtt_file = h5py.File(mtt_file)
    return np.asarray(mtt_file['mt']['t']).ravel()
