import numpy as np
import scipy

from multitrackpy import camera
from multitrackpy import geometry

def model3(abi,k):
    s = np.sqrt(np.sum(abi**2,axis=0))
    return abi * (1 + k[0]*s + k[1] * s**2)

def model3_inverse(ab_rd,k):
    n = ab_rd.shape[1]
    s = np.sqrt(np.sum(ab_rd**2,axis=0))
    r = np.zeros(n)

    for u in np.where(s > 0)[0]:
        rts = np.roots(np.array([k[1],k[0],1,-s[u]]))
        rtsind = np.all([np.imag(rts) == 0, rts >= 0],axis=0)
        if not np.any(rtsind):
            r[u] = np.nan
        else:
            r[u] = np.min(rts[rtsind])

    return ab_rd * (r / s)

def calc_space2image(xyz,offsets,w2cR,w2cT,camA,camk,sensorsize,scalepixels,discard=True,max_ab=False):
    if max_ab==False:
        max_ab = sensorsize
    
    xyz_cam = w2cR @ xyz + w2cT
    ab = xyz_cam[0:2,:] / xyz_cam[np.newaxis,2,:]
    ab = model3(ab,[0,camk[0]])
    ab = np.vstack((ab,np.ones((1,ab.shape[1]))));
    ab = camA @ ab
    ab = ab * scalepixels + (sensorsize +1)/2 - offsets

    if discard:
        ab = ab[:,np.any([ab[0,:] >= 1, ab[1,:] >= 1, ab[0,:] < max_ab[0], ab[1,:] < max_ab[1]],axis=0)]
        
    return ab

def calc_image2space(ab,offsets,w2cR,w2cT,camA,camk,sensorsize,scalepixels):
    ab = (ab - (sensorsize+1)/2 + offsets[:,np.newaxis]) / scalepixels;
    camAnull = scipy.linalg.null_space(camA)
    ab = np.linalg.lstsq(camA,ab,rcond=None)[0];
    ab = ab + camAnull/camAnull[2,:]*(1-ab[2,:])
    ab[0:2,:] = model3_inverse(ab[0:2,:],[0,camk[0,0]])
    ab = ab/np.sqrt(np.sum(ab**2,axis=0))
    return ((w2cR.T @ ab).T,(-w2cR.T @ w2cT).T)

def triangulate_points_nocorr(AB,offsets,calib,linedist_thres):
    n_AB = np.array([ ab.shape[0]  for ab in AB ])
    maincam = np.where(n_AB == np.max(n_AB[n_AB <= 12]))[0][0]
    #print(f'Maincam : {maincam}')
    cambases = np.empty((len(AB),3))

    full_ab = np.empty((len(AB),n_AB[maincam],2))
    full_ab[:] = np.NaN
    full_ab[maincam] = AB[maincam]

    full_dirs = np.empty((len(AB),n_AB[maincam],3))
    full_dirs[:] = np.NaN

    full_dirs[maincam,:,:],cambases[maincam,:] = camera.calc_image2space(AB[maincam].T,offsets[maincam],**calib[maincam]);

    for (iC,ab) in enumerate(AB):
        if iC == maincam: 
            continue
        if ab.shape[0] == 0:
            continue

        dirs,cambases[iC] = camera.calc_image2space(ab.T,offsets[iC],**calib[iC]);

        distances = np.array([ [ geometry.get_line_dist(cambases[maincam],full_dirs[maincam,i,:],cambases[iC],dirs[j]) for j in range(dirs.shape[0]) ] for i in range(full_dirs.shape[1]) ])
        
        np.equal(distances,np.min(distances,axis=0)[np.newaxis,:])
        connectmat = np.all([distances<linedist_thres,
                             np.equal(distances,np.min(distances,axis=1)[:,np.newaxis]), #np.equal(distances,np.min(distances,axis=0)[np.newaxis,:])
                            ],axis=0)
        connectmat[:,np.sum(connectmat,axis=0)>1] = False # Discard ambiguities
        corrs = np.array(np.where(connectmat))
        
        if corrs.shape[1] == 0:
            continue

        full_ab[iC,corrs[0],:] = ab[corrs[1]]
        #print(full_ab[iC])
        full_dirs[iC,corrs[0],:] = dirs[corrs[1].T]
    
    if full_dirs.shape[1] == 0:
        return np.zeros((0,3))
        
    points = np.array([ geometry.triangulate_3derr(cambases,full_dirs[:,i,:]) for i in range(full_dirs.shape[1]) ])
    #print(points)
    return points[~np.any(np.isnan(points),axis=1)]
