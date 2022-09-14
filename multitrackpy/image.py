# from scipy import ndimage
import numpy as np
import sys


def get_processed_frame(frame, kernel=None):
    # if kernel is None:
    #     kernel = [[0.1, 0.2, 0.1], [0.2, 1, 0.2], [.1, 0.2, 0.1]]
    # kernel = kernel/np.sum(kernel)
    # frame = ndimage.convolve(frame,kernel);

    return frame


def get_minima(framemap, led_thres=200):
    ledijs = np.array(np.where(framemap > led_thres))
    # print(f"Detecting minimum from {ledijs.shape[1]} thresholded pixels")
    if ledijs.shape[1] > 5000:
        print(f'Detected too many ({ledijs.shape[1]})threshold pixels, something is wrong. Exiting.', file=sys.stderr)
        exit()

    """localmax = framemap[ledijs[0],ledijs[1]] >= np.max(np.array([
     framemap[ledijs[0]+1,ledijs[1]],
     framemap[ledijs[0]-1,ledijs[1]],
     framemap[ledijs[0],ledijs[1]+1],
     framemap[ledijs[0],ledijs[1]-1],
     framemap[ledijs[0]+1,ledijs[1]+1],
     framemap[ledijs[0]-1,ledijs[1]-1],
     framemap[ledijs[0]-1,ledijs[1]+1],
     framemap[ledijs[0]+1,ledijs[1]-1],
    ]),axis=0)
    ledijs = np.array([ledijs[0][localmax],ledijs[1][localmax]])"""

    labelidx = 1
    lmlabels = np.zeros(ledijs.shape[1], dtype=np.int)

    for (i, lij) in enumerate(ledijs.T):
        for (j, offs) in enumerate(np.array([[1, 0], [1, -1], [1, 1], [0, 1]])):
            neigh = np.all(lij + offs == ledijs.T, axis=1)
            if np.any(neigh):
                if not lmlabels[neigh] == 0:
                    if lmlabels[i] == 0:
                        lmlabels[i] = lmlabels[neigh]
                    else:
                        lmlabels[lmlabels == lmlabels[i]] = lmlabels[neigh]

                if lmlabels[i] == 0:
                    lmlabels[i] = labelidx
                    labelidx = labelidx + 1

                lmlabels[neigh] = lmlabels[i]

    lmlabels[lmlabels == 0] = np.array(range(labelidx, labelidx + np.sum(lmlabels == 0)))

    ulabels = np.unique(lmlabels)
    minima = np.zeros((len(ulabels), 2))
    for i, label in enumerate(ulabels):
        weights = framemap[ledijs[0][lmlabels == label], ledijs[1][lmlabels == label], np.newaxis]
        minima[i] = (np.sum(weights * ledijs.T[lmlabels == label], axis=0) / np.sum(weights))

    return minima
