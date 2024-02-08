import numpy as np
import helper as hlp
import skimage.io as io
import submission as sub
import matplotlib.pyplot as plt
import skimage.io as skio

import open3d as o3d

# pts3d = np.load("../data/3d_points.npz")['pts3d']

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2], c='r', marker='o')
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# plt.show()
pnp = np.load('../data/pnp.npz', allow_pickle=True)
X = pnp['X']
x = pnp['x']
im = pnp['image']
cad = pnp['cad']

v = cad[0, 0][0]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(v[:, 0], v[:, 1], v[:, 2], c='r', marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
# # pcd = o3d.geometry.PointCloud()
# # pcd.points = o3d.utility.Vector3dVector(pts3d)
# # o3d.visualization.draw_geometries([pcd], zoom=0.3412, front=[0.4257, -0.2125, -0.8795])