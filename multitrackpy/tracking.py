import numpy as np
import imageio
from ccvtools import rawio
from multiprocessing import Pool
from itertools import starmap
from functools import partial

from multitrackpy import mtt
from multitrackpy import image
from multitrackpy import geometry
from multitrackpy import camera
from multitrackpy import helper


def track_frames(gopts):
    if gopts['n_cpu']>1:
        return track_frames_mp(gopts)
    else:
        return track_frames_sp(gopts)

def track_frames_sp(gopts,
                    space_coords=None, calib=None, videos=None, readers=None, offsets=None, 
                    R=None,t=None,errors=None,fr_out=None # E.g. for writing directly into slices of larger array
                    ):
    
    frame_idxs = gopts['frame_idxs']
    
    # Get inputs if not supplied
    if space_coords is None:
        space_coords = mtt.read_spacecoords(gopts['mtt_file'])
    if calib is None:
        calib = mtt.read_calib(gopts['mtt_file'])
    if videos is None:
        videos = mtt.read_video_paths(gopts['video_dir'],gopts['mtt_file']) 
    if readers is None:
        readers = [ imageio.get_reader(videos[i]) for i in range(len(videos)) ]
    if offsets is None:
        offsets = np.array([ reader.header['sensor']['offset'] for reader in readers ])
    
    if R is None:
        assert(t is None and errors is None and fr_out is None)
        R = np.empty((len(frame_idxs),3,3)); 
        t = np.empty((len(frame_idxs),3,1)); 
        errors = np.empty((len(frame_idxs),space_coords.shape[0])); 
        fr_out = np.empty((len(frame_idxs)),dtype=np.int32)
    else:
        assert(t is not None and errors is not None and fr_out is not None)
    
    # Initilize arrays
    R[:] = np.NaN
    t[:] = np.NaN
    errors[:] = np.NaN
    
    # Iterate frames for processing
    for (i,fr) in enumerate(frame_idxs):
        frames = np.array([ image.get_processed_frame(np.double(readers[iC].get_data(fr))) for iC in range(len(videos)) ])
        minima = [ np.flip(image.get_minima(frames[iC],gopts['led_thres']),axis=1) for iC in range(len(videos)) ] # minima return mat idxs, camera expects xy
        
        points = camera.triangulate_points_nocorr(minima,offsets,calib,gopts['linedist_thres'])
        
        fr_out[i] = fr
        
        if len(points)>0:
            R[i], t[i], errors[i] = geometry.find_trafo_nocorr(space_coords,points,gopts['corr_thres'])

    return (R,t,errors,fr_out)


def track_frames_mp(gopts):
    space_coords = mtt.read_spacecoords(gopts['mtt_file'])
    calib = mtt.read_calib(gopts['mtt_file'])
    videos = mtt.read_video_paths(gopts['video_dir'],gopts['mtt_file']) 
    print(f'Using {len(videos)} tracking cams')
    
    preloaddict = {
        'space_coords': space_coords,
        'calib': calib,
        'videos': videos,
        }
    
    frame_idxs = np.asarray(list(gopts['frame_idxs']))
    R = np.empty((len(frame_idxs),3,3)); R[:] = np.NaN
    t = np.empty((len(frame_idxs),3,1)); t[:] = np.NaN
    errors = np.empty((len(frame_idxs),space_coords.shape[0])); errors[:] = np.NaN
    fr_out = np.empty((len(frame_idxs)),dtype=np.int32)
    
    slice_list = list(helper.make_slices(len(frame_idxs),gopts['n_cpu']))
    arg_list = [ helper.dict_copyreplace(gopts,{'frame_idxs': frame_idxs[slice[0]:slice[1]]}) for slice in slice_list]
    
    with Pool(gopts['n_cpu']) as p:
        pres_list = p.map(partial(track_frames_sp,**preloaddict), arg_list)
    
    for (slice,pres) in zip(slice_list,pres_list): # Poolmap() returns in order
        R[slice[0]:slice[1]] = pres[0];
        t[slice[0]:slice[1]] = pres[1]
        errors[slice[0]:slice[1]] = pres[2]
        fr_out[slice[0]:slice[1]] = pres[3]

    return (R,t,errors,fr_out)


def get_default_globalopts():
    return {
        'mtt_file': '',
        'video_dir': '',
        'frame_idxs': None,
        'linedist_thres': 0.2, # Max distance between two cam lines to assume that the respective detections come from the same LED (calibration units)
        'corr_thres': 0.1, # Max diffeerence in point distance for correlation between model and detection (calibration units)
        'led_thres': 150, # Minimal brightness of LED after image processing (image brightness units)
        'n_cpu': 2,
        }


def check_globalopts(gopts):
    return (all(name in get_default_globalopts() for name in gopts) and
           all(name in gopts for name in get_default_globalopts()))