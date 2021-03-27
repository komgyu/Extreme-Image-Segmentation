import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy.interpolate as spint
import spline_rg as rg
from PIL import Image
from scipy import signal
import scipy.ndimage as nd

n = np.random.uniform(0.08, 0.92, 3600)  # generate 600 random points
data = np.array(n).reshape(400, 3, 3)  # 3 control points, 2 knots
spline = np.empty((200, 201, 3))  # 100splines, 101 output points, xyz value


# compute distance between 2 splines
def distance_two_splines(out_1, out_2):
    d_arr = np.array([40401])
    for w in range(201):
        for q in range(201):
            d = np.linalg.norm(out_1[w, :] - out_2[q, :])
            d_arr = np.append(d_arr, d)
    return np.amin(d_arr)

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
spline_number = 25

while i < 400 and t < spline_number:  # generate t splines
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
        if d_s[k] < 0.03:
            break
        #if d_s[k] > 0.05 and k < t - 1:
            #k += 1  #?????
        if k == t - 1 and d_s[k] > 0.03:
            spline[t, :, :] = out_arr #add to spline
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
ground_truth = np.full((100, 100, 100), 100, dtype=int)
colors = np.empty((100, 100, 100), dtype=object)
img = np.full((100, 100, 100), 1 , dtype = float)

for v in range(spline_number):
    for l in range(201):
        x = spline[v, l, 0]
        y = spline[v, l, 1]
        z = spline[v, l, 2]
        voxel[x, y, z] = True
        ground_truth[x, y, z] = v #cluster number for the ground truth, e.g. spline 0, spline 1, spline 2
        colors[x, y, z] = '0'
        img[x, y, z] = 0

        x = spline[v, l, 0] - 1
        y = spline[v, l, 1]
        z = spline[v, l, 2]
        voxel[x, y, z] = True
        ground_truth[x, y, z] = v
        colors[x, y, z] = '0'
        img[x, y, z] = 0

        x = spline[v, l, 0]
        y = spline[v, l, 1] - 1
        z = spline[v, l, 2]
        voxel[x, y, z] = True
        ground_truth[x, y, z] = v
        colors[x, y, z] = '0'
        img[x, y, z] = 0

        x = spline[v, l, 0]
        y = spline[v, l, 1]
        z = spline[v, l, 2] - 1
        voxel[x, y, z] = True
        ground_truth[x, y, z] = v
        colors[x, y, z] = '0'
        img[x, y, z] = 0

        x = spline[v, l, 0] + 1
        y = spline[v, l, 1]
        z = spline[v, l, 2]
        voxel[x, y, z] = True
        ground_truth[x, y, z] = v
        colors[x, y, z] = '0'
        img[x, y, z] = 0

        x = spline[v, l, 0]
        y = spline[v, l, 1] + 1
        z = spline[v, l, 2]
        voxel[x, y, z] = True
        ground_truth[x, y, z] = v
        colors[x, y, z] = '0'
        img[x, y, z] = 0

        x = spline[v, l, 0]
        y = spline[v, l, 1]
        z = spline[v, l, 2] + 1
        voxel[x, y, z] = True
        ground_truth[x, y, z] = v
        colors[x, y, z] = '0'
        img[x, y, z] = 0
        

        # colors = np.empty(voxels.shape, dtype=object)
        # colors[voxels] = '0.5'
print("111111")




def smooth(colors, img):
    Next = [ [-1, -1, -1],[-1, 0, -1],
             [1, 1, 1], [1, 2, 1],   
             [0, -1, 0], [1, -1, 1], [1, 1, -3],
             [-1, 1, 0], [-1, -1, 0],
             [-1, 0, 3], [-2, 0, 0],
             [0, 1, 0],[-1, 0, -4], 
             [0, 0,  3], [0, 3, 0], [0, -3, 0]
             ]
    
    nv = np.random.uniform(0, 100, 60000).astype(int)
    nv = nv.reshape((20000, 3))

    for n in range(20000):
        x = nv[n, 0]
        y = nv[n, 1]
        z = nv[n, 2]
        if voxel[x, y, z] == True:
                for differ in Next:
                    xiv = x + differ[0]
                    yiv = y + differ[1]
                    ziv = z + differ[2]
                    if -1< xiv< 100  and -1< yiv< 100 and -1< ziv< 100:
                        if voxel[xiv, yiv, ziv] !=True:
                            colors[xiv, yiv, ziv] = '0.3'
                            img[xiv, yiv, ziv] = 0.3                      
    
    return colors, img


# add noise to the background 
def add_noise(colors, img):
    noise = np.random.uniform(2, 98, 1500).astype(int)
    noise = np.array(noise).reshape((500, 3))
    for n in range(500):
        [vx, vy, vz] = noise[n].tolist()
        colors[vx, vy, vz] = '0.7'
        img[vx, vy, vz] = 0.7
      
    return colors, img

colors, img = smooth(colors, img)
colors, img = add_noise(colors, img)


for vx in range(100):
    for vy in range(100):
        for vz in range(100):
            if voxel[vx, vy, vz] == True:   #splines are set true
                img[vx, vy, vz] = 0 # 0 means the the pixel is spline, 0 means nothing, 0.5 means noise
                colors[vx, vy, vz] = '0'
               
            if voxel[vx, vy, vz] == False:
                ground_truth[vx, vy, vz] = 100 # labeling clusters which do not belong to splines
                if 0< img[vx, vy, vz] < 1: 
                    voxel[vx, vy, vz] = True 
                else: 
                    img[vx, vy, vz] = 1
                    colors[vx, vy, vz] = '1'


for x in range(100):
    for y in range(100):
        for z in range(100):
            voxel[x, y, z] = False

for x in range(100):
    for y in range(100):
        #for z in range(50):
            voxel[x, y, 50] = True

for x in range(100):
    #for y in range(50):
        for z in range(100):
            voxel[x, 50, z] = True

#fig, axs = plt.subplots(2)
fig = plt.figure(figsize = plt.figaspect(0.5)) # set up a figure twice as wide as it is tall
#fig = plt.figure()
ax1 = fig.add_subplot(1,2,1, projection = '3d')
ax1.title.set_text('Ground Truth')
ax1.voxels(voxel, facecolors= colors)
#----------------------------------------------------------------

#cluster = np.full((100, 100, 100), 100 , dtype = int)  #used to store the clustering results

#outimg = rs.regionGrowing(img, seed, 0.2)
outImg, cluster = rg.regionGrowing(img, ground_truth, spline_number)

voxel_output = np.full((100, 100, 100), False, dtype=bool)

print ("voxel_output")

#for x in range(100):
    #for y in range(100):
        #for z in range(100):
            #if outImg[x, y, z] == 0:
                #voxel_output[x, y, z] = True

for x in range(100):
    for y in range(100):
        #for z in range(50):
            if outImg[x, y, 50] == 0:
                voxel_output[x, y, 50] = True
                colors[x, y, 50] = '0'
            if outImg[x, y, 50] == 1:
                voxel_output[x, y, 50] = True
                colors[x, y, 50] = '1'

for x in range(100):
    #for y in range(50):
        for z in range(100):
            if outImg[x, 50, z] == 0:
                voxel_output[x, 50, z] = True
                colors[x, 50, z] = '0'
            if outImg[x, 50, z] == 1:
                voxel_output[x, 50, z] = True
                colors[x, 50, z] = '1'




ax2 = fig.add_subplot(1,2,2, projection = '3d')
ax2.title.set_text('Decomposition Result')
ax2.voxels(voxel_output, facecolors= colors )


plt.show()


