import numpy as np
import h5py


def read_calib(mtt_path):
    mtt_file = h5py.File(mtt_path)

    istracking = np.squeeze(np.asarray([mtt_file['mt']['cam_istracking']]) == 1)
    calind = np.squeeze(np.int32(mtt_file['mt']['calind']))[istracking] - 1
    calibs = []

    for ci in calind:
        w2cR = np.squeeze(
            np.asarray(mtt_file['mt']['mc']['Rglobal'])[ci])  # The way matlab saves arrays alread includes a permute!
        w2cT = np.asarray((w2cR @ -np.asarray(mtt_file['mt']['mc']['Tglobal'])[ci].T))

        scaling = np.asarray(mtt_file[mtt_file['mt']['mc']['cal']['scaling'][ci, 0]]).T[0]
        icent = np.asarray(mtt_file[mtt_file['mt']['mc']['cal']['icent'][ci, 0]]).T[0]
        camA = np.asarray([[scaling[0], scaling[2], icent[0]],
                           [0, scaling[1], icent[1]]])

        camk = np.asarray(mtt_file[mtt_file['mt']['mc']['cal']['distortion_coefs'][ci, 0]])
        sensorsize = np.asarray(mtt_file[mtt_file['mt']['mc']['cal']['sensorsize'][ci, 0]])
        scalepixels = np.asarray(mtt_file[mtt_file['mt']['mc']['cal']['scale_pixels'][ci, 0]])

        calibs.append({'w2cR': w2cR, 'w2cT': w2cT, 'camA': camA, 'camk': camk, 'sensorsize': sensorsize,
                       'scalepixels': scalepixels})

    return calibs


def read_video_paths(vid_dir, mtt_path):
    mtt_file = h5py.File(mtt_path)
    istracking = np.squeeze(np.asarray([mtt_file['mt']['cam_istracking']]) == 1)
    return [vid_dir + ''.join([chr(c) for c in mtt_file[mtt_file['mt']['vidname'][0, i]][:].T.astype(np.int)[0]]) for i
            in np.where(istracking)[0]]


def read_spacecoords(mtt_path):
    mtt_file = h5py.File(mtt_path)
    return np.asarray(mtt_file['mt']['objmodel']['space_coord'])


def read_frame_n(mtt_path):
    mtt_file = h5py.File(mtt_path)
    return len(mtt_file['mt']['t'])
