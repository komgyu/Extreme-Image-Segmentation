import numpy as np

def regionGrowing(grayImg, outImg, seeds, threshold, cluster, c):
    """
    :param grayImg: gray images
    :param seed: starting point for growing
    :param threshold: 
    :return: 
    """
    [maxX, maxY,maxZ] =  grayImg.shape[0:3]
    seed = seeds[0]
    seed = np.array([seed[0], seed[1], seed[2]])
    print(seed)

    seeds = np.delete(seeds, 0, 0) #delete seeds[0] because it is used
   
    # queue for storing growing points
    pointQueue = []
    pointQueue.append((seed[0], seed[1],seed[2]))
  
    outImg[seed[0], seed[1],seed[2]] = 0

    pointsNum = 1
    pointsMean = float(grayImg[seed[0], seed[1],seed[2]])

    # 6 neighbors
    Next6 = [[-1, 0, 0],[1, 0, 0], 
              [0, 1, 0], [0, -1, 0],
              [0, 0, 1], [0, 0, -1]]
    print ("start")
    p = 0
    while(len(pointQueue)>0):
         # get the first point and delete
        growSeed = pointQueue[0]
        del pointQueue[0]

        for differ in Next6:
            growPointx = growSeed[0] + differ[0]
            growPointy = growSeed[1] + differ[1]
            growPointz = growSeed[2] + differ[2]

            # if it is boundary point
            if((growPointx < 0) or (growPointx > maxX - 1) or
               (growPointy < 0) or (growPointy > maxY - 1) or (growPointz < 0) or (growPointz > maxZ - 1)) :
                continue

             # if it is growed
            if(outImg[growPointx,growPointy,growPointz] == 1):
                continue

            data = grayImg[growPointx,growPointy,growPointz]
            
            # condition satisfied, add to growing list
            if(abs(data-pointsMean)<threshold):
                pointsNum += 1
                pointsMean = (pointsMean * (pointsNum - 1) + data) / pointsNum
                outImg[growPointx, growPointy,growPointz] = 1
                cluster[growPointx, growPointy,growPointz] = c
                pointQueue.append([growPointx, growPointy,growPointz])
               
            
            for i in range(seeds.shape[0]):
                if growPointx==seeds[i, 0] and growPointy==seeds[i, 1] and growPointz==seeds[i, 2]:
                    seeds = np.delete(seeds, i, 0)
                    break
    print("end")
   
    return outImg, cluster, seeds

