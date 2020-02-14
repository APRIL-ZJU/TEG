import numpy as np
import cv2
import scipy.ndimage.filters as filters
from scipy import ndimage
from skimage.feature import peak_local_max


def detect_edge(gray, k=2 ** (1.0 / 3)):
    # k>=1, threshold variable is the contrast threshold

    pyrlvl1 = ndimage.filters.gaussian_filter(gray, 1.6 * k)
    pyrlvl2 = ndimage.filters.gaussian_filter(gray, 1.6 * (k ** 2))
    pyrlvl3 = ndimage.filters.gaussian_filter(gray, 1.6 * (k ** 4))

    # Difference-of-Gaussians (DoG)
    diff1 = abs(pyrlvl2 - pyrlvl1)
    diff2 = abs(pyrlvl3 - pyrlvl2)
    diff = diff1 + diff2
    return diff


def map_depth_to_cloud(depth, Kmat, MM_PER_M=1000.0):
    cx = Kmat[0][2]
    cy = Kmat[1][2]
    fx = Kmat[0][0]
    fy = Kmat[1][1]

    h, w = depth.shape

    # convert depth image to 3D point clouds
    xgrid = np.ones([h, 1])*range(w) - cx       # 480*640, xgrid[i][:] = 0~640
    s = np.reshape(np.arange(h), [1, h])
    ygrid = np.transpose(s)*np.ones([w]) - cy   # 480*640, ygrid[:][i] = 0~480

    pc = np.zeros([h, w, 3])
    pc[:, :, 0] = xgrid*depth/fx/MM_PER_M
    pc[:, :, 1] = ygrid*depth/fy/MM_PER_M
    pc[:, :, 2] = depth/MM_PER_M
    return pc


def normalization(data, vmax=10, vmin=0):
    mmax = np.max(data)
    mmin = np.min(data)
    k = vmax/(mmax - mmin)
    newdata = vmin + k*(data - mmin)
    return newdata


def self_correlation(img, sigma=0.3, alpha=0.04):
    Ix, Iy = np.gradient(img)
    H11 = filters.gaussian_filter(Ix * Ix, sigma)   # kernel = 2*int(4*sigma+0.5)+1 = 10
    H12 = filters.gaussian_filter(Ix * Iy, sigma)
    H22 = filters.gaussian_filter(Iy * Iy, sigma)
    dt = H11 * H22 - H12 * H12  # dt is the determinant
    tr = H11 + H22              # tr is the trace
    Re = dt - alpha * (tr ** 2)
    return Re


def select_keypoint(R, ksize=4, s=50, weight=1e-4, num=None):
    # select the local maximum
    coors = peak_local_max(R, min_distance=ksize)

    # filter
    h, w = R.shape
    thre = weight * R.max()
    kps = [[x[0], x[1], R[x[0]][x[1]]] for x in coors if R[x[0]][x[1]] > thre and s < x[0] < h-s and s < x[1] < w-s]

    if num is not None and len(kps) > num:
        fkps = sorted(sorted(kps), key=lambda x:x[2], reverse=True)
        res = fkps[:num]
    else:
        res = kps

    fres = [[x[0], x[1]] for x in res]
    return fres


def detector(grayImage, depthImage, Kmat, belt=1, num=400, ksize=3, border=50, weight=1e-4):
    '''
    :param grayImage: int, [480, 640]
    :param depthImage: int, [480, 640]
    :param Kmat: float, [4,4], the intrinsics matrix of the RGB-D camera
    :param belt: int, the weight of R_rgb compared to R_depth
    :param num: the number of needed keypoints, the output number is <= num
    :param ksize: when getting local maximum, the size of windows = ksize * 2 + 1
    :param border: the

    :return: kps: int, [N, 2], the coordinates of keypoints
    '''

    # extract DOG information
    gray = np.double(grayImage)          # NOTE: it's necessary for the gaussian filter
    diff = detect_edge(gray)

    diff = normalization(diff)
    R_rgb = abs(self_correlation(diff, sigma=0.3))

    # process depth image
    depthImage = filters.gaussian_filter(depthImage, 1)
    pc = map_depth_to_cloud(depthImage, Kmat)

    # extract gradient information
    dx, dy, dz = np.gradient(pc)
    gra = abs(dx[:, :, 0]) + abs(dx[:, :, 1]) + abs(dy[:, :, 0]) + abs(dy[:, :, 1])
            # + abs(dx[:, :, 2]) + abs(dy[:, :, 2])

    gra = normalization(gra)
    R_depth = abs(self_correlation(gra, sigma=0.8))

    # filter the point whose depth is missing
    mask = depthImage > 1
    R_depth *= mask

    # extract kps
    R = belt*R_rgb + R_depth
    kps = select_keypoint(R, ksize=ksize, s=border, weight=weight, num=num)
    return kps


if __name__ == '__main__':
    import time

    Kmat = np.loadtxt('./data/ManipulatorsDataset/camera-intrinsics.txt')
    rgbfile1 = './data/ManipulatorsDataset/mixture/illum/rgb_1.png'
    depthfile1 = './data/ManipulatorsDataset/mixture/illum/depth_1.png'
    rgbImage1 = cv2.imread(rgbfile1, 1)
    grayImage1 = cv2.imread(rgbfile1, 0)
    depthImage1 = cv2.imread(depthfile1, cv2.IMREAD_UNCHANGED)

    time0 = times.times()
    kps1 = detector(grayImage1, depthImage1, Kmat)
    print(times.times() - time0)
    print(len(kps1))
