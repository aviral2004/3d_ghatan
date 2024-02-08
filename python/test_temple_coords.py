import numpy as np
import helper as hlp
import skimage.io as io
import submission as sub
import matplotlib.pyplot as plt
import skimage.io as skio

# 1. Load the two temple images and the points from data/some_corresp.npz
im1 = skio.imread('../data/im1.png')
im2 = skio.imread('../data/im2.png')

corresp = np.load('../data/some_corresp.npz')
pts1 = corresp['pts1']
pts2 = corresp['pts2']

# 2. Run eight_point to compute F
M = max(im1.shape)
F = sub.eight_point(pts1, pts2, M)

# hlp.displayEpipolarF(im1, im2, F)

# 3. Load points in image 1 from data/temple_coords.npz
corresp = np.load("../data/temple_coords.npz")
pts1 = corresp['pts1']

# hlp.epipolarMatchGUI(im1, im2, F)

# 4. Run epipolar_correspondences to get points in image 2
pts2 = sub.epipolar_correspondences(im1, im2, F, pts1)

# 5. Compute the camera projection matrix P1
intrinsics = np.load("../data/intrinsics.npz")
K1 = intrinsics['K1']
K2 = intrinsics['K2']

P1 = np.hstack((K1, np.zeros((3, 1))))

# 6. Use camera2 to get 4 camera projection matrices P2
E = sub.essential_matrix(F, K1, K2)
P2s = hlp.camera2(E)

# 7. Run triangulate using the projection matrices
def pts_in_front(P1, P2, pts3d):
    pts3d = np.vstack((pts3d.T, np.ones((1, pts3d.shape[0]))))
    tmp = ((P1 @ pts3d)[2] > 0) & ((P2 @ pts3d)[2] > 0)
    # count trues in total
    return np.sum(tmp)

i = 3
P2 = K2 @ P2s[:, :, i]
pts3d = sub.triangulate(P1, pts1, P2, pts2)
print(f"total points: {pts3d.shape[0]}")
print(pts_in_front(P1, P2, pts3d))

np.savez("../data/3d_points.npz", pts3d=pts3d[:, :3])
# 8. Figure out the correct P2


# 9. Scatter plot the correct 3D points
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2], c='r', marker='o')
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# plt.show()

# 10. Save the computed extrinsic parameters (R1,R2,t1,t2) to data/extrinsics.npz
R1 = np.eye(3)
t1 = np.zeros((3, 1))
R2 = P2s[:, :, i][:, :3]
t2 = P2s[:, :, i][:, 3].reshape(-1, 1)
np.savez("../data/extrinsics.npz", R1=R1, R2=R2, t1=t1, t2=t2)