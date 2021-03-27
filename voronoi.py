import numpy as np
import matplotlib.pyplot as plt
import voronoi_rg as rg

point = np.random.uniform(0, 50, 300).astype(int)
sites = np.array(point).reshape(100, 3)



# set the colors of each object


# assign every voxel to a certain cluster
def generate_voxel():
    voxel = np.full((50, 50, 50), 500, dtype = int)
    for x in range(50):
        for y in range(50):
            for z in range(50):
                d = []
                for n in range(100):
                   d = np.append(d, np.linalg.norm(sites[n] - [x, y, z]))    
                min_value = min(d)
                voxel[x, y, z] = d.tolist().index(min_value)
    return voxel
    


#merge from n into m voronoi cells, actually (m-1) small voronoi cells
def merge_voronoi(n_before, m_after, voxel): 
    ground_truth = np.full((50, 50, 50), 100, dtype = int) #ground truth stores the number for every voronoi, 100 means boundary
    voxel_merged = voxel
    d = []
    #merged_index = []

    #for k in range(m_after-1):
        #merged_index = np.append(merged_index, k)

    # the cells are not necessarily convex, merge cell 0 and cell i, i >=m
    for i in range(m_after, n_before, 1):  
        d = np.append(d, np.linalg.norm(sites[0]- sites[i]))
    min_value = min(d)
    index_min = d.tolist().index(min_value)
    for x in range(50):
        for y in range(50):
            for z in range(50):
                if voxel[x, y, z] == index_min + m_after:
                    voxel_merged[x, y, z] = 0
                    ground_truth[x, y, z] = 0 #cell number = 0
    

    # merge the rest voronoi cells into one 
    for i in range(m_after, n_before, 1):
        for x in range(50):
            for y in range(50):
                for z in range(50):
                    if voxel_merged[x, y, z] == i:
                        voxel_merged[x, y, z] = m_after -1
                        ground_truth[x, y, z] = m_after -1
       
    return voxel_merged, ground_truth




#color the voronoi cell

def colorize(m, voxel): 
    voxel_color = np.full((50, 50, 50), False, dtype = bool)
    colors = np.empty((50, 50, 50), dtype=object)
    color = np.random.uniform(0, 1.0, 600)
    color_array = np.array(color).reshape(200, 3)

    for x in range(50):
        for y in range(50):
            for z in range(50):
                for n in range(m):
                    if voxel[x, y, z] == n:
                        voxel_color[x, y, z] = True
                        colors[x, y, z] = color_array[n].tolist()
    return voxel_color, colors


#check boundary, if the voxel is boundary it is black
def check_boundary(nv, voxel, ground_truth): #if voxel = 50*50*50, nv = 50-1
  
    voxel_boundary = np.full((50, 50, 50), False, dtype = bool) 
    img = np.full((50, 50, 50), 1, dtype = float)
    colors = np.empty((50, 50, 50), dtype=object)
    #check 8 points
    if voxel[0, 0, 0]!= voxel[1, 0, 0] or voxel[0, 0, 0]!= voxel[0, 1, 0] or voxel[0, 0, 0]!= voxel[0, 0, 1]:
        voxel_boundary[0, 0, 0]  = True
    if voxel[nv, nv, nv] != voxel[nv -1, nv, nv] or voxel[nv, nv, nv] != voxel[nv, nv-1, nv] or voxel[nv, nv, nv] != voxel[nv, nv, nv-1]:
        voxel_boundary[nv, nv, nv] = True
    if voxel[0, nv, nv] != voxel[1, nv, nv] or voxel[0, nv, nv] != voxel[0, nv-1, nv] or voxel[0, nv, nv] != voxel[0, nv, nv-1]:
        voxel_boundary[0, nv, nv] = True
    if voxel[nv, 0, 0] != voxel[nv-1, 0, 0] or voxel[nv, 0, 0] != voxel[0, 1, nv] or voxel[0, 0, nv] != voxel[0, 0, nv-1]:
        voxel_boundary[nv, 0, 0] = True
    if voxel[0, 0, nv] != voxel[1, 0, nv] or voxel[0, 0, nv] != voxel[0, 1, nv] or voxel[0, 0, nv] != voxel[0, 0, nv-1]:
        voxel_boundary[0, 0, nv] = True
    if voxel[nv, nv, 0] != voxel[nv-1, nv, 0] or voxel[nv, nv, 0] != voxel[nv, nv-1, 0] or voxel[nv, nv, 0] != voxel[nv, nv, 1]:
        voxel_boundary[nv, nv, 0] = True
    if voxel[0, nv, 0] != voxel[1, nv, 0] or voxel[0, nv, 0] != voxel[0, nv-1, 0] or voxel[0, nv, 0] != voxel[0, nv, 1]:
        voxel_boundary[0, nv, 0] = True
    if voxel[nv, 0, nv] != voxel[nv-1, 0, nv] or voxel[nv, 0, nv] != voxel[nv, 1, nv] or voxel[nv, 0, nv] != voxel[nv, 0, nv-1]:
        voxel_boundary[nv, 0, nv] = True

    #check 12 lines
    for z in range (1, nv, 1):
        if voxel[0, 0, z] != voxel[1, 0, z] or voxel[0, 0, z] != voxel[0, 1, z] or voxel[0, 0, z] != voxel[0, 0, z+1] or voxel[0, 0, z] != voxel[0, 0, z-1]:
            voxel_boundary[0, 0, z] = True
        if voxel[0, nv, z] != voxel[1, nv, z] or voxel[0, nv, z] != voxel[0, nv-1, z] or voxel[0, nv, z] != voxel[0, nv, z+1] or voxel[0, nv, z] != voxel[0, nv, z-1]:
            voxel_boundary[0, nv, z] = True
        if voxel[nv, 0, z] != voxel[nv-1, 0, z] or voxel[nv, 0, z] != voxel[nv, 1, z] or voxel[nv, 0, z] != voxel[nv, 0, z+1] or voxel[nv, 0, z] != voxel[nv, 0, z-1]:
            voxel_boundary[nv, 0, z] = True
        if voxel[nv, nv, z] != voxel[nv-1, nv, z] or voxel[nv, nv, z] != voxel[nv, nv-1, z] or voxel[nv, nv, z] != voxel[nv, nv, z+1] or voxel[nv, nv, z] != voxel[nv, nv, z-1]:
            voxel_boundary[nv, nv, z] = True
    
    for y in range (1, nv, 1):
        if voxel[0, y, 0] != voxel[1, y, 0] or voxel[0, y, 0] != voxel[0, y, 1] or voxel[0, y, 0] != voxel[0, y+1, 0] or voxel[0, y, 0] != voxel[0, y-1, 0]:
            voxel_boundary[0, y, 0] = True
        if voxel[0, y, nv] != voxel[1, y, nv] or voxel[0, y, nv] != voxel[0, y, nv-1] or voxel[0, y, nv] != voxel[0, y+1, nv] or voxel[0, y, nv] != voxel[0, y-1, nv]:
            voxel_boundary[0, y, nv] = True
        if voxel[nv, y, 0] != voxel[nv-1, y, 0] or voxel[nv, y, 0] != voxel[nv, y, 1] or voxel[nv, y, 0] != voxel[nv, y+1, 0] or voxel[nv, y, 0] != voxel[nv, y-1, 0]:
            voxel_boundary[nv, y, 0] = True
        if voxel[nv, y, nv] != voxel[nv-1, y, nv] or voxel[nv, y, nv] != voxel[nv, y, nv-1] or voxel[nv, y, nv] != voxel[nv, y+1, nv] or voxel[nv, y, nv] != voxel[nv, y-1, nv]:
            voxel_boundary[nv, y, nv] = True
    
    for x in range (1, nv, 1):
        if voxel[x, 0, 0] != voxel[x, 1, 0] or voxel[x, 0, 0] != voxel[x, 0, 1] or voxel[x, 0, 0] != voxel[x+1, 0, 0] or voxel[x, 0, 0] != voxel[x-1, 0, 0]:
            voxel_boundary[x, 0, 0] = True
        if voxel[x, 0, nv] != voxel[x, 1, nv] or voxel[x, 0, nv] != voxel[x, 0, nv-1] or voxel[x, 0, nv] != voxel[x+1, 0, nv] or voxel[x, 0, nv] != voxel[x-1, 0, nv]:
            voxel_boundary[x, 0, nv] = True
        if voxel[x, nv, 0] != voxel[x, nv, 1] or voxel[x, nv, 0] != voxel[x, nv-1, 0] or voxel[x, nv, 0] != voxel[x+1, nv, 0] or voxel[x, nv, 0] != voxel[x-1, nv, 0]:
            voxel_boundary[x, nv, 0] = True
        if voxel[x, nv, nv] != voxel[x, nv-1, nv] or voxel[x, nv, nv] != voxel[x, nv, nv-1] or voxel[x, nv, nv] != voxel[x+1, nv, nv] or voxel[x, nv, nv] != voxel[x-1, nv, nv]:
            voxel_boundary[x, nv, nv] = True
            
    
    #check 6 surfaces
    #x = 0
    for y in range(1, nv, 1):
        for z in range(1, nv, 1):
            a = voxel[0, y, z] != voxel[1, y, z] or voxel[0, y, z] != voxel[0, y+1, z] or voxel[0, y, z] != voxel[0, y-1, z]
            b = voxel[0, y, z] != voxel[0, y, z+1] or voxel[0, y, z] != voxel[0, y, z-1]
            if (a or b)== True:
                voxel_boundary[0, y, z] = True
                
    #x = nv
    for y in range(1, nv, 1):
        for z in range(1, nv, 1):
            a = voxel[nv, y, z] != voxel[nv-1, y, z] or voxel[nv, y, z] != voxel[nv, y+1, z] or voxel[nv, y, z] != voxel[nv, y-1, z]
            b = voxel[nv, y, z] != voxel[nv, y, z+1] or voxel[nv, y, z] != voxel[nv, y, z-1]
            if (a or b) == True:
                voxel_boundary[nv, y, z] = True
        
    #y = 0
    for x in range(1, nv, 1):
        for z in range(1, nv, 1):
            a = voxel[x, 0, z] != voxel[x, 1, z] or voxel[x, 0, z] != voxel[x+1, 0, z] or voxel[x, 0, z] != voxel[x-1, 0, z]
            b = voxel[x, 0, z] != voxel[x, 0, z+1] or voxel[x, 0, z] != voxel[x, 0, z-1]
            if (a or b) == True:
                voxel_boundary[x, 0, z] = True
                
    #y = nv
    for x in range(1, nv, 1):
        for z in range(1, nv, 1):
            a = voxel[x, nv, z] != voxel[x, nv-1, z] or voxel[x, nv, z] != voxel[x+1, nv, z] or voxel[x, nv, z] != voxel[x-1, nv, z]
            b = voxel[x, nv, z] != voxel[x, nv, z+1] or voxel[x, nv, z] != voxel[x, nv, z-1]
            if (a or b)==True:
                voxel_boundary[x, nv, z] = True
                
    #z = 0
    for x in range(1, nv, 1):
        for y in range(1, nv, 1):
            a = voxel[x, y, 0] != voxel[x, y, 1] or voxel[x, y, 0] != voxel[x+1, y, 0] or voxel[x, y, 0] != voxel[x-1, y, 0]
            b = voxel[x, y, 0] != voxel[x, y+1, 0] or voxel[x, y, 0] != voxel[x, y-1, 0]
            if (a or b)==True:
                voxel_boundary[x, y, 0] = True
              
    
    #z = nv
    for x in range(1, nv, 1):
        for y in range(1, nv, 1):
            a = voxel[x, y, nv] != voxel[x, y, nv-1] or voxel[x, y, nv] != voxel[x+1, y, nv] or voxel[x, y, nv] != voxel[x-1, y, nv]
            b = voxel[x, y, nv] != voxel[x, y+1, nv] or voxel[x, y, nv] != voxel[x, y-1, nv]
            if (a or b) == True:
                voxel_boundary[x, y, nv] = True 
               

   #check inside volume (50-1)**3
    for x in range(1, nv, 1): 
        for y in range(1, nv, 1):
            for z in range(1, nv, 1):
                a = voxel[x, y, z] != voxel[x+1, y, z] or voxel[x, y, z] != voxel[x-1, y, z] or voxel[x, y, z] != voxel[x, y+1, z]
                b = voxel[x, y, z] != voxel[x, y-1, z] or voxel[x,y, z]!= voxel[x, y, z+1] or voxel[x, y, z] != voxel[x, y, z-1]
                if (a or b)== True:
                   voxel_boundary[x, y, z]  = True
                  
 
    for x in range(nv+1): 
        for y in range(nv+1):
            for z in range(nv+1):
                if voxel_boundary[x, y, z] == True:
                    colors[x, y, z] = '0.1'
                    img[x, y, z] = 0
                    ground_truth[x, y, z] = 100


                if voxel_boundary[x, y, z] == False:
                     voxel_boundary[x, y, z] = True
                     colors[x, y, z] = '1'
                     img[x, y, z] = 1

             
    
    
    return voxel_boundary, colors, img, ground_truth

#pixels near boundary are black
def smooth(voxel, colors, img):
    Next = [ [-1, -1, -1], [1, 0, 0],
             [1, 1, 1], [1, 2, 1],   
             [0, -1, 0], [1, -1, 1], [1, 1, -3],
             [-1, 1, 0], [-1, -1, 0],
             [-2, 0, 0], [-1, 0, 0],
             [0, 1, 0], [0, 2, 0],
             [0, 0,  3]
             ]
    
    nv = np.random.uniform(0, 50, 30000).astype(int)
    nv = nv.reshape((10000, 3))

    for n in range(10000):
        x = nv[n, 0]
        y = nv[n, 1]
        z = nv[n, 2]
        if voxel[x, y, z] == True:
                for differ in Next:
                    xiv = x + differ[0]
                    yiv = y + differ[1]
                    ziv = z + differ[2]
                    if -1< xiv< 50 and -1< yiv< 50 and -1< ziv< 50:
                        if voxel[xiv, yiv, ziv] !=True:
                            voxel[xiv, yiv, ziv] = True
                            colors[xiv, yiv, ziv] = '0.2'
                            img[xiv, yiv, ziv] = 0.2                      
    
    return voxel, colors, img

def add_noise(voxel, colors, img):
    voxel_noise = voxel
    noise = np.random.uniform(0, 50, 2400).astype(int)
    noise = np.array(noise).reshape((800, 3))
    for n in range(800):
        [vx, vy, vz] = noise[n].tolist()
        colors[vx, vy, vz] = '0.7'
        img[vx, vy, vz] = 0.8
        voxel[vx, vy, vz] = True

    return voxel, colors, img
    



# and plot everything


fig = plt.figure(figsize = plt.figaspect(0.33))

voxel = generate_voxel()

print(voxel[5, 45, 6])
print(voxel[5, 36, 6])

ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax1.title.set_text('Original Voronoi')
voxel_color, colors = colorize(100, voxel)
ax1.voxels(voxel_color, facecolors=colors)


cell = 10
voxel_merged, ground_truth = merge_voronoi(100, cell, voxel)

#voxel_color2, colors2 = colorize(4, voxel_merged)
voxel, colors, img, ground_truth = check_boundary(49, voxel_merged, ground_truth)

voxel, colors, img = smooth(voxel, colors, img)
voxel, colors, img = add_noise(voxel, colors, img)
for x in range(50):
    for y in range(50):
        for z in range(50):
            voxel[x, y, z] = False

for x in range(50):
    for y in range(50):
        #for z in range(50):
            voxel[x, y, 25] = True

for x in range(50):
    #for y in range(50):
        for z in range(50):
            voxel[x, 25, z] = True

ax2 = fig.add_subplot(1, 3, 2, projection='3d') 
ax2.title.set_text('Ground Truth')
ax2.voxels(voxel, facecolors = colors)


                


#---------------------------------------------------------
#---------------------------------------------------------
outImg, cluster = rg.regionGrowing(img, ground_truth, cell)

voxel_output = np.full((50, 50, 50), False, dtype=bool)

print ("voxel_output")

for x in range(50):
    for y in range(50):
        #for z in range(50):
            if outImg[x, y, 25] == 0:
                voxel_output[x, y, 25] = True
                colors[x, y, 25] = '0'
            if outImg[x, y, 25] == 1:
                voxel_output[x, y, 25] = True
                colors[x, y, 25] = '1'

for x in range(50):
    #for y in range(50):
        for z in range(50):
            if outImg[x, 25, z] == 0:
                voxel_output[x, 25, z] = True
                colors[x, 25, z] = '0'
            if outImg[x, 25, z] == 1:
                voxel_output[x, 25, z] = True
                colors[x, 25, z] = '1'


ax3 = fig.add_subplot(1, 3, 3, projection='3d')
ax3.title.set_text('Decomposition Result')
ax3.voxels(voxel_output, facecolors= colors )

plt.show()
            


            



                    
                


                    

                

            
