import numpy as np
import submission as sub
import matplotlib.pyplot as plt

# write your implementation here
pnp = np.load('../data/pnp.npz', allow_pickle=True)
cad = pnp['cad']
x = pnp['x']
X = pnp['X']
im = pnp['image']

P = sub.estimate_pose(x, X)
K, R, t = sub.estimate_params(P)

vertices = cad[0, 0][0]
proj_v = P @ np.vstack((vertices.T, np.ones(vertices.shape[0])))
proj_v = proj_v / proj_v[-1, :]
proj_v = proj_v[:2, :].T

plt.imshow(im)
# reduce opacity of the lines
plt.plot(proj_v.T[0], proj_v.T[1], 'b-', alpha=0.5)
plt.show()