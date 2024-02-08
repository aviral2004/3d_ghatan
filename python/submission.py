"""
Homework 5
Submission Functions
"""

# import packages here
import numpy as np
import helper as hlp

"""
Q3.1.1 Eight Point Algorithm
       [I] pts1, points in image 1 (Nx2 matrix)
           pts2, points in image 2 (Nx2 matrix)
           M, scalar value computed as max(H1,W1)
       [O] F, the fundamental matrix (3x3 matrix)
"""
def eight_point(pts1, pts2, M):
    # normalise points by M
    # T = np.diag([1/M, 1/M])
    # pts1 = (T @ pts1.T).T
    # # pts2 = (T @ pts2.T).T
    # pts1 = pts1/M
    # pts2 = pts2/M

    # form correspondence matrix
    A = np.zeros((pts1.shape[0], 9))
    for i in range(pts1.shape[0]):
        x, y = pts1[i]
        _x, _y = pts2[i]
        A[i] = [_x*x, _x*y, _x, _y*x, _y*y, _y, x, y, 1]

    # SVD of A, F is the column with the smallest singular value
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3,3)

    # Enforce rank 2 constraint
    U, S, Vt = np.linalg.svd(F)
    # S[2] = 0
    F = U @ np.diag([*S[:2], 0]) @ Vt

    # T = np.array([[1/M,0,0],[0,1/M,0],[0,0,1]])
    # F = T.T @ F @ T
    # F = hlp.refineF(F, pts1, pts2)

    return F

"""
Q3.1.2 Epipolar Correspondences
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           F, fundamental matrix from image 1 to image 2 (3x3 matrix)
           pts1, points in image 1 (Nx2 matrix)
       [O] pts2, points in image 2 (Nx2 matrix)
"""
def sad(patch1, patch2):
    return np.sum(np.abs(patch1 - patch2))

def ssd(patch1, patch2):
    return np.sum((patch1 - patch2)**2)

def ncc(patch1, patch2):
    return np.sum((patch1 - patch1.mean()) * (patch2 - patch2.mean())) / (patch1.std() * patch2.std())

def epipolar_correspondences(im1, im2, F, pts1):
    pts2 = np.zeros_like(pts1)

    win_size = 10

    for i, p1 in enumerate(pts1):
        _x1, _y1 = p1
        win1 = im1[_y1-win_size:_y1+win_size, _x1-win_size:_x1+win_size]

        l = F @ np.array([_x1, _y1, 1])
        a, b, c = l
        x = np.arange(0, im2.shape[1])
        y = (-c - a*x) / b
        y = np.round(y).astype(int)
        
        best_score = 0
        best_x2, best_y2 = 0, 0
        for _x2, _y2 in zip(x, y):
            if _x2-win_size < 0 or _x2+win_size >= im2.shape[1] or _y2-win_size < 0 or _y2+win_size >= im2.shape[0]:
                continue

            win2 = im2[_y2-win_size:_y2+win_size, _x2-win_size:_x2+win_size]
            score = ncc(win1, win2)

            if score > best_score:
                best_score = score
                best_x2, best_y2 = _x2, _y2

        pts2[i] = [best_x2, best_y2]

    return pts2


"""
Q3.1.3 Essential Matrix
       [I] F, the fundamental matrix (3x3 matrix)
           K1, camera matrix 1 (3x3 matrix)
           K2, camera matrix 2 (3x3 matrix)
       [O] E, the essential matrix (3x3 matrix)
"""
def essential_matrix(F, K1, K2):
    return K2.T @ F @ K1


"""
Q3.1.4 Triangulation
       [I] P1, camera projection matrix 1 (3x4 matrix)
           pts1, points in image 1 (Nx2 matrix)
           P2, camera projection matrix 2 (3x4 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] pts3d, 3D points in space (Nx3 matrix)
"""
def triangulate(P1, pts1, P2, pts2):
    def gen_mat(P, pt):
        p1, p2, p3 = P
        x, y = pt
        return np.array([
            y*p3 - p2,
            p1 - x*p3
        ])

    pts3d = np.zeros((pts1.shape[0], 3))
    for i in range(pts1.shape[0]):
        A = np.vstack((gen_mat(P1, pts1[i]), gen_mat(P2, pts2[i])))
        _, _, Vt = np.linalg.svd(A)
        pts3d[i] = Vt[-1, :3] / Vt[-1, 3]

    return pts3d


"""
Q3.2.1 Image Rectification
       [I] K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] M1 M2, rectification matrices (3x3 matrix)
           K1p K2p, rectified camera matrices (3x3 matrix)
           R1p R2p, rectified rotation matrices (3x3 matrix)
           t1p t2p, rectified translation vectors (3x1 matrix)
"""
def rectify_pair(K1, K2, R1, R2, t1, t2):
    c1 = -np.linalg.inv(K1 @ R1) @ (K1 @ t1) # 3x1
    c2 = -np.linalg.inv(K2 @ R2) @ (K2 @ t2)

    r1 = (c2 - c1) / np.linalg.norm(c2 - c1) # 3x1
    r2 = np.cross(R1[2], r1.T).T # 3x1
    r3 = np.cross(r1.T, r2.T).T #3x1

    R1p = np.hstack((r1, r2, r3)).T
    R2p = R1p

    K1p = K2
    K2p = K2

    t1p = -R1p @ c1
    t2p = -R2p @ c2

    M1 = (K1p @ R1p) @ np.linalg.inv(K1 @ R1)
    M2 = (K2p @ R2p) @ np.linalg.inv(K2 @ R2)

    return M1, M2, K1p, K2p, R1p, R2p, t1p, t2p

"""
Q3.2.2 Disparity Map
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           max_disp, scalar maximum disparity value
           win_size, scalar window size value
       [O] dispM, disparity map (H1xW1 matrix)
"""
def get_disparity(im1, im2, max_disp, win_size):
    im1pad = np.pad(im1, win_size//2, mode='constant')
    im2pad = np.pad(im2, win_size//2, mode='constant')

    print(im1.shape)
    print(im2.shape)
    print(im1pad.shape)
    dispM = np.zeros_like(im1)

    for x in range(im1.shape[1]):
        for y in range(im1.shape[0]):
            x = x + win_size//2
            y = y + win_size//2
            win1 = im1pad[y-win_size//2:y+win_size//2, x-win_size//2:x+win_size//2]
            best_score = np.inf
            best_disp = 0
            d_range = min(max_disp, im1pad.shape[1]-x)
            for d in range(d_range):
                win2 = im2pad[y-win_size//2:y+win_size//2, x-win_size//2+d:x+win_size//2+d]
                # print(win1.shape, win2.shape)
                score = ssd(win1, win2)
                if score < best_score:
                    best_score = score
                    best_disp = d
            x = x-win_size//2
            y = y-win_size//2
            dispM[y, x] = best_disp

    return dispM

"""
Q3.2.3 Depth Map
       [I] dispM, disparity map (H1xW1 matrix)
           K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] depthM, depth map (H1xW1 matrix)
"""
def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    f = K1[1, 1]
    c1 = -np.linalg.inv(K1 @ R1) @ (K1 @ t1) # 3x1
    c2 = -np.linalg.inv(K2 @ R2) @ (K2 @ t2)

    b = np.linalg.norm(c2 - c1)

    depthM = np.zeros_like(dispM)
    depthM[dispM != 0] = b * f / dispM[dispM != 0]

    return depthM


"""
Q3.3.1 Camera Matrix Estimation
       [I] x, 2D points (Nx2 matrix)
           X, 3D points (Nx3 matrix)
       [O] P, camera matrix (3x4 matrix)
"""
def estimate_pose(x, X):
    def get_mat(x, X):
        _X, _Y, _Z = X
        u, v = x
        return np.array([
            [_X, _Y, _Z, 1, 0, 0, 0, 0, -u*_X, -u*_Y, -u*_Z, -u],
            [0, 0, 0, 0, _X, _Y, _Z, 1, -v*_X, -v*_Y, -v*_Z, -v]
        ])

    A = np.vstack([get_mat(x, X) for x, X in zip(x, X)])
    _, _, Vt = np.linalg.svd(A.T @ A)
    P = Vt[-1].reshape(3,4)
    return P

"""
Q3.3.2 Camera Parameter Estimation
       [I] P, camera matrix (3x4 matrix)
       [O] K, camera intrinsics (3x3 matrix)
           R, camera extrinsics rotation (3x3 matrix)
           t, camera extrinsics translation (3x1 matrix)
"""
def estimate_params(P):
    _, _, V = np.linalg.svd(P)
    c = (V[-1]/V[-1, -1])[:3]

    K, R = np.linalg.qr(P[:, :3])
    t = -R@c

    return K, R, t
