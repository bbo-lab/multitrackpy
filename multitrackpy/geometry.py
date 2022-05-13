import numpy as np
import scipy
from .rigid_body_transform_3d import rigid_transform_3D


def get_line_dist(r1, e1, r2, e2):
    n = np.cross(e1, e2)
    return np.abs(np.sum(n * (r1 - r2)))


def triangulate_3derr(bases, vecs):
    p = np.empty(3)
    p[:] = np.NaN

    rayok = ~np.any([
        np.isnan(bases),
        np.isnan(vecs)
    ], axis=(0, 2))

    bases = bases[rayok]
    vecs = vecs[rayok]

    n = bases.shape[0]
    if n < 2:
        return p

    M = np.empty((n, 3, 3))
    Mbase = np.empty((n, 3, 1))

    for u in range(n):
        planebasis = scipy.linalg.null_space(vecs[np.newaxis, u])
        M[u] = planebasis @ planebasis.T
        Mbase[u] = M[u] @ bases[u, np.newaxis].T

    if np.linalg.matrix_rank(np.sum(M, axis=0)) < 3:
        return p

    return np.squeeze(np.linalg.solve(np.sum(M, axis=0), np.sum(Mbase, axis=0)).T)


def find_trafo_nocorr(pc1, pc2, corr_thres):
    errors = np.empty(pc1.shape[0])
    errors[:] = np.NaN

    corrs = find_correspondences(pc1, pc2, corr_thres)

    if not np.any(np.equal(corrs.shape, 0)):
        (R, t) = rigid_transform_3D(pc1[corrs[0]].T, pc2[corrs[1]].T)
        errors[corrs[0]] = np.sqrt(np.sum(((R @ pc1[corrs[0]].T + t) - pc2[corrs[1]].T) ** 2, axis=0))
    else:
        R = np.empty((3, 3))
        R[:] = np.NaN
        t = np.empty((3, 1))
        t[:] = np.NaN

    return R, t, errors


def calc_dists(points, idx):
    return np.sqrt(np.sum((points - points[idx]) ** 2, axis=1))


def find_correspondences(p1, p2, corr_thres=0.1):
    p1dists = [np.sort(calc_dists(p1, i)) for i in range(p1.shape[0])]
    p2dists = [np.sort(calc_dists(p2, i)) for i in range(p2.shape[0])]
    disterr_n = np.zeros((len(p1dists), len(p2dists)))
    for (i, p1d) in enumerate(p1dists):
        for (j, p2d) in enumerate(p2dists):
            for d in p2d:
                if d != 0 and np.min(np.abs(p1d - d)) < corr_thres:
                    disterr_n[i, j] = disterr_n[i, j] + 1

    corrmat = np.where(np.all([np.equal(disterr_n, np.max(disterr_n, axis=0)[np.newaxis, :]),
                               disterr_n > p2.shape[0] * 0.6], axis=0))
    corrs = np.asarray(corrmat)

    # Discard ambiguities
    uniquemask = [np.sum(corrs[0] == c) == 1 for c in corrs[0]]
    corrs = corrs.T[uniquemask].T
    return corrs
