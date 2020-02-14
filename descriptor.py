from sklearn.preprocessing import scale
from collections import Counter
import scipy.ndimage.filters as filters
import numpy as np
import cv2


def map_depth_to_pc(depth, Kmat, MM_PER_M=1000.0):
    cx = Kmat[0][2]
    cy = Kmat[1][2]
    fx = Kmat[0][0]
    fy = Kmat[0][0]

    depth = np.double(depth)            # 480*640
    h, w = depth.shape

    # construct kinect map
    depthMask = depth > 0.0

    # convert depth image to 3D point clouds
    xgrid = np.ones([h, 1])*range(w) - cx       # 480*640, xgrid[i][:] = 0~640
    s = np.reshape(np.arange(h), [1, h])
    ygrid = np.transpose(s)*np.ones([w]) - cy   # 480*640, ygrid[:][i] = 0~480

    pc = np.zeros([h, w, 3])
    pc[:, :, 0] = xgrid*depth/fx/MM_PER_M
    pc[:, :, 1] = ygrid*depth/fy/MM_PER_M
    pc[:, :, 2] = depth/MM_PER_M
    return pc, depthMask


def fit_plane_for_pc(data):
    '''
    Fit a plane to the set of coordinates.

    For a passed list of points in (x,y,z) cartesian coordinates,
    find the plane that best fits the data, the unit vector normal to that plane
    with an initial point at the average of the x, y, and z values.
    '''

    flag = np.zeros(3)
    datadim = [[] for _ in range(3)]
    n = [[] for _ in range(3)]
    for i in range(3):
        X = [[1., x[1], x[2]] for x in data]

        # check wherther x_m can be solved
        X_m = np.dot(np.transpose(X), X)  # X' * X = |X|, 3*3
        if np.linalg.det(X_m) == 0:  # X_m is reversible
            flag[i] = 0
            continue
        else:
            flag[i] = 1

        # Construct and normalize the normal vector
        a = np.dot(np.linalg.pinv(X_m), np.transpose(X))
        datadim[i] = [data[j][i] for j in range(len(data))]
        coeff = np.dot(a, datadim[i])
        c_neg = -coeff
        c_neg[i] = 1
        coeff[i] = 1
        n[i] = c_neg / np.linalg.norm(coeff)  # 3*3

    if sum(flag) == 0:
        return 0, 0
    else:
        # Calculating residuals for each fit
        off_center = [datadim[0] - np.mean(datadim[0]), datadim[1] - np.mean(datadim[1]), datadim[2] - np.mean(datadim[2])]

        residual_sum = np.zeros(3)
        for i in range(3):
            if flag[i] == 0:
                residual_sum[i] = None
                continue

            residuals = np.dot(np.transpose(off_center), n[i])
            residual_sum[i] = sum(residuals ** 2)

        # find the lowest index
        residual_sum = residual_sum.tolist()
        best_fit = residual_sum.index(min(residual_sum))
        nor = n[best_fit]
        return 1, nor


def get_feature(patch_gray, patch_grad, patch_dp, mask, ngray, ngrad, ndp, ps):
    def get_idx(patch, nblock):
        patch = np.float64(patch)
        patch = cv2.resize(patch, (ps, ps))

        # if minimal<0, translate values > 0 since we use the relative order
        mini = patch.min()
        if mini <= 0:
            trans = max(int(np.ceil(abs(mini))), 1)  # >= 1
            patch += trans

        # turn square patch to circle
        patch *= mask

        # compute blocks' gaps for circle patch
        val = np.unique(patch)
        mini = val[1]       # minimal val[0] is 0 from mask
        maxi = val[-1]
        gap = ((maxi - mini) / nblock) + 1e-6

        # get the idx of values whose mask is True
        divs = np.floor((patch - mini) / gap) + 1  # move ids >= 1 to differ from points whose mask is False

        divs *= mask
        divs = divs.flatten().astype(np.int).tolist()
        divs = list(filter(lambda x: x > 0, divs))  # only save the points whose mask is True
        return divs

    # get idx for each patch
    idx_gray = get_idx(patch_gray, ngray)
    idx_grad = get_idx(patch_grad, ngrad)
    idx_dp = get_idx(patch_dp, ndp)

    cat = np.stack([idx_gray, idx_grad, idx_dp], axis=1)
    cat = cat.astype(np.str).tolist()  # turn to string for hash
    cat = [' '.join(x) for x in cat]
    cou = Counter(cat)  # statistic

    # build histogram
    hist = np.zeros([ngray, ngrad, ndp])
    for key in cou.keys():
        id1, id2, id3 = key.split(' ')
        hist[int(id1) - 1, int(id2) - 1, int(id3) - 1] = cou[key]

    desp = hist.flatten()
    return desp


def descriptor(grayImage, depthImage, kps, Kmat, ngray=20, ngrad=20, ndp=20, r=20, ps=40, sr=25, bg_thre=0.3):
    '''
    :param grayImage: int, [480, 640]
    :param depthImage: int, [480, 640]
    :param kps: int, [N,2], the coordinates of keypoints
    :param Kmat: float, [4,4], the intrinsics matrix of the RGB-D camera
    :param ngray: int, the number of gray blocks.
    :param ngrad: int, the number of pointcloud gradient blocks
    :param ndp: int, the number of dotpoint spatial blocks
    :param r: int, the size of initial patch.
    :param ps: int, the size of final patch. All patches are unified to [ps, ps]
    :param sr: int, a radius used to select neighboring points
    :param bg_thre: float, a threshold to check whether a point is a background one or not in neighbors

    :return: kps: int, [N',2], the coordinates of keypoints
    :return: desp: float, [N',D], the feature vectors of keypoints
    '''

    # the image preprocessing
    h, w = grayImage.shape
    gray = filters.gaussian_filter(grayImage, 1)
    depth = filters.gaussian_filter(depthImage, 1)

    pc, depthMask = map_depth_to_pc(depth, Kmat)

    # generate a mask to turn square patch to circle patch
    center = int(ps // 2)
    patchMask = cv2.circle(np.zeros([ps, ps]), (center, center), center, (1, 1, 1), -1)

    keypoints = []
    features = []
    for k in range(len(kps)):
        u, v = kps[k]
        x, y, z = pc[u, v]

        if depthMask[u, v]:   # if the depth of this keypoint exists
            # estimate scale according to the empirical equation in BRAND
            if 2 < z < 8:
                s = max(0.2, (3.8-0.4*max(2.0, z))/3)   # scale
                r = int(round(r * s))                   # modify the radius of initial patch

            # filter distant points to get neighbors
            minu = max(u - sr, 0)
            maxu = min(u + sr, h)
            minv = max(v - sr, 0)
            maxv = min(v + sr, w)
            patch = pc[minu:maxu, minv:maxv, :]
            patch_mask = depthMask[minu:maxu, minv:maxv]

            urange, vrange, _ = patch.shape
            npoints = urange * vrange
            patch_points = patch.reshape([npoints, 3])

            neigh_m = abs(patch[:, :, 2] - z) < bg_thre    # according to z-axis
            dep_m = patch_mask != 0                     # depth exists
            neigh_m = (neigh_m * dep_m).flatten()
            neighbors = [patch_points[i, :] for i in range(npoints) if neigh_m[i]]

            if len(neighbors) >= 50: #2 * r**2:
                try:
                    # get the dominant orientation
                    flag, normal = fit_plane_for_pc(neighbors)
                    if flag != 0:    # if the plane of pc can be fit
                        n_w = [[normal[0]+x], [normal[1]+y], [normal[2]+z]]

                        # get the patch to be encoded
                        minu = max(u - r, 0)
                        maxu = min(u + r, h)
                        minv = max(v - r, 0)
                        maxv = min(v + r, w)
                        patch_pc = pc[minu:maxu, minv:maxv, :]

                        ## get gray patch
                        patch_gray = gray[minu:maxu, minv:maxv]

                        ## get spatial dotproduct patch
                        points = patch_pc.reshape([(maxu-minu)*(maxv-minv), 3])
                        patch_dp = np.dot(points, n_w)
                        patch_dp = patch_dp.reshape([maxu-minu, maxv-minv])

                        ## get pc_grad patch
                        dx, dy, dz = np.gradient(patch_pc)
                        patch_grad = abs(dx[:, :, 0]) + abs(dx[:, :, 1]) + abs(dy[:, :, 0]) + abs(dy[:, :, 1]) + abs(dx[:, :, 2]) + abs(dy[:, :, 2]) + abs(dz[:, :, 1]) + abs(dz[:, :, 2])

                        desp = get_feature(patch_gray, patch_grad, patch_dp, patchMask, ngray, ngrad, ndp, ps)

                        keypoints.append([u, v])
                        features.append(desp)
                except:
                    continue

    feats = scale(features, axis=0, with_mean=True, with_std=True)   # normalize
    return keypoints, feats


if __name__ == '__main__':
    import time
    from detector import *

    pathdir = './data/ManipulatorsDataset/'
    Kmat = np.loadtxt(pathdir + 'camera-intrinsics.txt')
    rgbfile = pathdir + 'mixture/illum/rgb_1.png'
    depthfile = pathdir + 'mixture/illum/depth_1.png'
    rgbImage = cv2.imread(rgbfile)
    grayImage = cv2.imread(rgbfile, 0)
    depthImage = cv2.imread(depthfile, cv2.IMREAD_UNCHANGED)

    kps = detector(grayImage, depthImage, Kmat)
    print(len(kps))

    time1 = time.time()
    kps, desp = descriptor(grayImage, depthImage, kps, Kmat)
    print(time.time() - time1)
    print(len(kps))
