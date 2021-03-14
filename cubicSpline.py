import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy.interpolate as spint
#import regiongrowing.py

n = np.random.uniform(0.08, 0.92, 2700)  # generate 600 random points
data = np.array(n).reshape(300, 3, 3)  # 3 control points, 2 knots
spline = np.empty((200, 201, 3))  # 100splines, 101 output points, xyz value


# compute distance between 2 splines
def distance_two_splines(out_1, out_2):
    d_arr = np.array([100])
    for w in range(201):
        for q in range(201):
            d = np.linalg.norm(out_1[w, :] - out_2[q, :])
            d_arr = np.append(d_arr, d)
    return np.amin(d_arr)


plt.figure()
ax = plt.axes(projection='3d')
# ax.set_xlim(0.0, 1.0)
# ax.set_ylim(0.0, 1.0)
# ax.set_zlim(0.0, 1.0)

new_data = data[0, :, :]
tck, u = interpolate.splprep(new_data.transpose(), k=2)
unew = np.arange(0, 1.005, 0.005)
out = interpolate.splev(unew, tck)
out_arr = np.array(out).transpose()  # one output spline
spline[0, :, :] = out_arr
# ax.plot3D(out[0], out[1], out[2], color='black')
# ax.scatter(new_data.transpose()[0, :], new_data.transpose()[1, :], new_data.transpose()[2, :])

i = 0
t = 1

while i < 200 and t < 25:  # generate 100 splines
    new_data = data[i + 1, :, :]
    tck, u = interpolate.splprep(new_data.transpose(), k=2)
    unew = np.arange(0, 1.005, 0.005)
    out = interpolate.splev(unew, tck)
    out_arr = np.array(out).transpose()  # one output spline

    i += 1
    k = 0
    d_s = np.zeros(shape=t)
    for k in range(t):
        d_s[k] = distance_two_splines(spline[k, :, :], out_arr)
        if d_s[k] < 0.0:
            break
        if d_s[k] > 0.02 and k < t - 1:
            # print(d_s[k])
            k += k
        if k == t - 1 and d_s[k] > 0.02:
            spline[t, :, :] = out_arr
            spline_transpose = spline[t, :, :].transpose()
            # ax.plot3D(spline_transpose[0], spline_transpose[1], spline_transpose[2], color='0.8')
            # ax.scatter(new_data.transpose()[0, :], new_data.transpose()[1, :], new_data.transpose()[2, :])
            t += 1
            print(t)
            # print(i)
            break
print("success")

# ax.plot3D(out[0], out[1], out[2])
# ax.scatter(new_data[:, 0], new_data[:, 1], new_data[:, 2])
# x, y, z = np.indices(100, 100, 100)
spline = (spline * 100).astype(int)

voxel = np.full((100, 100, 100), False, dtype=bool)

for v in range(25):
    for l in range(201):
        x = spline[v, l, 0]
        y = spline[v, l, 1]
        z = spline[v, l, 2]
        voxel[x, y, z] = True

        x = spline[v, l, 0] - 1
        y = spline[v, l, 1]
        z = spline[v, l, 2]
        voxel[x, y, z] = True
        x = spline[v, l, 0]
        y = spline[v, l, 1] - 1
        z = spline[v, l, 2]
        voxel[x, y, z] = True
        x = spline[v, l, 0]
        y = spline[v, l, 1]
        z = spline[v, l, 2] - 1
        voxel[x, y, z] = True
        x = spline[v, l, 0] + 1
        y = spline[v, l, 1]
        z = spline[v, l, 2]
        voxel[x, y, z] = True
        x = spline[v, l, 0]
        y = spline[v, l, 1] + 1
        z = spline[v, l, 2]
        voxel[x, y, z] = True
        x = spline[v, l, 0]
        y = spline[v, l, 1]
        z = spline[v, l, 2] + 1
        voxel[x, y, z] = True

        # colors = np.empty(voxels.shape, dtype=object)
        # colors[voxels] = '0.5'
print("111111")
ax.voxels(voxel, facecolors='0')  # edgecolor='k')

plt.show()
