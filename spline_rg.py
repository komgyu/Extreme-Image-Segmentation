import numpy as np
import region_grow_s as rs
import sklearn.metrics as sk
import math


def getNeighbors(seed):
    neighbors = np.empty((6, 3), dtype = int)
    neighbors[0] = np.array([seed[0] + 1, seed[1],     seed[2]]    ) 
    neighbors[1] = np.array([seed[0],     seed[1] + 1, seed[2]]    ) 
    neighbors[2] = np.array([seed[0],     seed[1],     seed[2] + 1]) 
    neighbors[3] = np.array([seed[0] - 1, seed[1],     seed[2]]    )
    neighbors[4] = np.array([seed[0],     seed[1] - 1, seed[2]]    )
    neighbors[5] = np.array([seed[0],     seed[1],     seed[2] - 1])
    neighbors = np.array(neighbors)
    return neighbors


def regionGrowing(img, ground_truth, spline_number):
    
    i = 0 # possible seed number
    cluster = np.full((100, 100, 100), 100 , dtype = int)  #voxels = 100 means it does not belong to 
    seeds = np.empty((0, 3), int)
    #choose all possible seeds where voxel color is black
    for x in range(100):
        for y in range(100):
            for z in range(100):
                if img[x, y, z] == 0:
                    seeds = np.append(seeds, np.array([[x,y,z]]), axis=0)
                    


    print(seeds.shape)
    i = seeds.shape[0]

    print(seeds[3, 0])

    # get real seeds, delete seeds which are not suitable 

    new_seed = np.empty((0, 3), int)

    for n in range(i):
        if 0 < seeds[n, 0] < 99 and 0 < seeds[n, 1] < 99 and 0 < seeds[n, 2] < 99:
            neighbors = getNeighbors(seeds[n])
            value = 0
            for m in range(6):
                value = value +img[neighbors[m, 0], neighbors[m, 1], neighbors[m, 2]]
            if value/6 < 0.16: # six neighbors are all black, allowing one noisy point with 1/6
                new_seed = np.append(new_seed, np.array([[seeds[n, 0],seeds[n, 1],seeds[n, 2]]]), axis=0)
    
    
    seeds = new_seed
    print(seeds.shape)

    outImg = np.full((100, 100, 100), 1 , dtype = int)
    
    m = 0
    while seeds.shape[0] > 3:  

        outImg, cluster, seeds = rs.regionGrowing(img,outImg, seeds, 0.2, cluster, m )
        m += 1
        print(m)


    # compute RI
    RI = sk.rand_score(ground_truth.reshape((100**3)), cluster.reshape((100**3)))
    print (RI)

    # compute VI
    def compute_VI(m, cluster, spline_number, ground_truth):
        n = 100**3
        r = np.full((m+1, spline_number+1), 0 , dtype = int)
        clusterVI = np.full((m+1), 0, dtype = int)
        groundVI = np.full((spline_number+1), 0, dtype = int)
        for x in range(100):
            for y in range(100):
                for z in range(100):
                    i = cluster[x, y, z]
                    j = ground_truth[x, y, z]
                    if i ==100 and j!=100:
                        r[m, j] = r[m, j] +1
                        clusterVI[m] = clusterVI[m] + 1
                        groundVI[j] = groundVI[j] + 1

                    if j == 100 and i != 100:
                        r[i,spline_number] = r[i, spline_number] + 1
                        clusterVI[i] = clusterVI[i] + 1
                        groundVI[spline_number] = groundVI[spline_number] + 1
                    
                    if i!= 100 and j!=100:
                        r[i,j] = r[i, j] + 1
                        clusterVI[i] = clusterVI[i] + 1
                        groundVI[j] = groundVI[j] + 1
        

        VI = 0
        for i in range(m+1):
            for j in range(spline_number+1):
                if r[i, j] >0:
                    VI = VI - r[i, j]/n*(math.log(r[i,j]/clusterVI[i]) + math.log(r[i,j]/groundVI[j]))
        
        return VI
    
    VI = compute_VI(m, cluster, spline_number, ground_truth)
    print(VI)

    return outImg, cluster




    



                



                
    


